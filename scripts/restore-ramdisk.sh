#!/bin/bash
# Refill the RAM disk after a power cycle (run on the Mac): tmpfs on the server, a one-off tar|nc
# stream from the NAS, then the Q8_0-dense variant next to hard links of the untouched shards.
#   restore-ramdisk.sh [server]        (default r820.local)
# The model is ~474 GiB: about 70-75 min over 1 GbE. The first ik run with GGML_NUMA_PLACE=1 then
# moves every row quarter to its node (~4 min, persists until the next power-off).
set -e
SRV=${1:-r820.local}
NAS=nas.local
SRC=/volume1/@home/richard/models/DeepSeek-V4.1-Flash-1017-GGUF
S() { ssh -i ~/.ssh/id_rsa -o BatchMode=yes -o ServerAliveInterval=15 richard@"$SRV" "$@"; }

IP=$(S 'hostname -I | cut -d" " -f1')
echo "server $SRV at $IP"

# huge=never from the start: expert row quarters (1.6 MB) are smaller than a 2 MB page, and 4 KiB
# pages read just as fast here; no splitpages.py pass needed
S 'set -e; sudo mkdir -p /mnt/models
   mountpoint -q /mnt/models || sudo mount -t tmpfs -o size=560G,mpol=interleave,huge=never,uid=1000,gid=1000,mode=0755 tmpfs /mnt/models
   findmnt -no OPTIONS /mnt/models
   nohup sh -c "nc -l 9000 | tar xf - -C /mnt/models" > ~/bench/recv.log 2>&1 &
   sleep 1; ss -ltn | grep -q ":9000 " && echo "listener up"'

ssh -o BatchMode=yes "$NAS" "cd $SRC && nohup sh -c 'tar cf - UD-Q8_K_XL | nc -N $IP 9000' > /tmp/send-q8.log 2>&1 &"
echo "stream started; waiting for it to finish"

until ! S 'pgrep -x tar >/dev/null'; do
    S 'du -sh /mnt/models 2>/dev/null' | tr '\n' ' '; date +%T
    sleep 60
done
S 'ls -l /mnt/models/UD-Q8_K_XL/ | awk "{print \$5, \$NF}"'

# BF16 dense weights -> Q8_0 (only the last shard holds them), hard links for the rest
S 'set -e; D=/mnt/models/dq8; S=/mnt/models/UD-Q8_K_XL; mkdir -p $D
   for i in 1 2 3 4 5 6 7 8; do f=DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-0000$i-of-00009.gguf; [ -e $D/$f ] || ln $S/$f $D/$f; done
   OMP_NUM_THREADS=32 ~/bench/requant_shard $S/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00009-of-00009.gguf \
       $D/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00009-of-00009.gguf q8_0 token_embd.weight=keep | tail -n 1'

# state that does not survive a reboot
S 'sudo nvidia-smi -pm 1 >/dev/null; sudo modprobe msr; numactl -H | grep free'
echo "done: /mnt/models/dq8 is ready"
