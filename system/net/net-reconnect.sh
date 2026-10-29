#!/bin/sh
# Booted without a network cable (it is often on the iDRAC port)? Once a minute, re-run DHCP on
# every wired interface that has carrier, until one of them gives us a default route; then exit.
while :; do
    if ip -4 route show default | grep -q .; then
        logger -t net-reconnect "connected: $(ip -4 -o addr show scope global | awk '{print $2, $4}' | tr '\n' ' ')"
        exit 0
    fi
    for dev in /sys/class/net/*; do
        i=${dev##*/}
        case $i in lo|docker*|virbr*|veth*|br-*) continue ;; esac
        [ -e "$dev/device" ] || continue                       # physical NICs only
        [ "$(cat "$dev/carrier" 2>/dev/null)" = 1 ] || continue
        logger -t net-reconnect "carrier on $i but no default route: reconfiguring"
        networkctl reconfigure "$i" 2>/dev/null || netplan apply
    done
    sleep 60
done
