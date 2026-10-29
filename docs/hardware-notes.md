# Hardware notes

## Boot failure (VLT0204)

The server refused to POST with VLT0204. The cause was the front control panel cable (CTRL_PNL on the board). It stays unplugged.

Side effect: **the front power button does nothing.** Power the server on and off through the iDRAC (`racadm serveraction powerup` / `graceshutdown`, or the web UI).

## Network

Only one ethernet cord is used. It's moved between the iDRAC dedicated port and the server's first onboard NIC as needed, so the iDRAC and the OS are never reachable at the same time.
- The OS gets its address over DHCP and announces itself as `r820.local` (avahi). netplan uses `dhcp-identifier: mac`, so the lease stays stable across reboots.
- After moving the cord to the server NIC, the OS may need a reboot with the cable already in to pick up a lease. Reboot through the iDRAC and swap the cord during POST.
- The NIC must stay on the dedicated iDRAC port; don't switch it to shared LOM.

## Power cap

The server arrived with a **433 W DCMI power limit** active, probably from a previous owner. Under memory load it throttled the cores to ~1,800 MHz and made bandwidth erratic. Removed in-band:

```bash
sudo ipmitool dcmi power get_limit
sudo ipmitool dcmi power deactivate
```

Full-load draw is now ~630 W with cores at ~2,600 MHz.

## BIOS (2.7.0, the last release)

Only one change was needed: System Profile from PerfPerWattOptimizedDapc to **PerfOptimized** (C-states and C1E off, processor power MaxPerf, turbo on). Already right: memory at 1333 (the ceiling for 3 dual-rank DIMMs per channel at 1.5 V), Optimizer mode, node interleave off, QPI at maximum, prefetchers on.

Memory speed: 3 DIMMs per channel caps it at 1333. 2 per channel (32 sticks, 512 GB) allows 1600 (+20% bandwidth), but 512 GB is too small for the Q8 download.

### Changing BIOS attributes through the iDRAC (racadm over SSH)

```bash
racadm set BIOS.SysProfileSettings.SysProfile PerfOptimized
racadm jobqueue create BIOS.Setup.1-1 -s TIME_NOW
racadm serveraction graceshutdown && racadm serveraction powerup
```

### Stuck BIOS jobs (LC068 / RAC1038)

A job stuck in "Automated Task Execution", or LC068 / RAC1038 "a configuration job is already committed" for a job that isn't in the queue, means a ghost job. This sequence cleared it:

```bash
racadm racreset hard        # wait for the iDRAC to come back (several minutes)
racadm jobqueue delete -i JID_CLEARALL_FORCE
racadm set ...              # set the attribute again
racadm jobqueue create BIOS.Setup.1-1 -s TIME_NOW
racadm serveraction graceshutdown && racadm serveraction powerup
```

- `racadm clearpending <FQDD>` aborts pending (uncommitted) attributes for one device, e.g. `RAID.Slot.7-1`.
- Setting `LifecycleController.LCAttributes.CollectSystemInventoryOnRestart` to Disabled made the first BIOS job stall; leave it enabled if jobs misbehave.
- `idrac/abort-bios-pending.xml` is the WS-Man `DCIM_BIOSService.DeletePendingConfiguration` request, as a last resort.

## GPU

The Quadro P2200 sits on NUMA node 1, PCIe 3.0 x16. Driver: `nvidia-headless-580-server` (580.178.04, closed module; the open modules are Turing+ only), CUDA 12.4 from Ubuntu's `nvidia-cuda-toolkit`, gcc-13 as the nvcc host compiler.
- Nouveau had bound the card first: unbind through sysfs, then `system/modprobe.d/blacklist-nouveau.conf`.
- Ubuntu's `71-nvidia.rules` remove rules (`modprobe -r`) ping-pong with the add rules when the driver is loaded live. `system/udev/71-nvidia.rules` is an override without them.
- Persistence mode (`nvidia-smi -pm 1`) doesn't survive a reboot yet.

## Open hardware items

- PSU1 reports AC lost (check the cord or the PSU).
- CMOS battery.
- Control panel cable or panel repair, to get the power button back.
