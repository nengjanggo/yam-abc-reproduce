#!/usr/bin/env bash
# Sequential USB-CAN binding for YAM leaders and followers.
set -euo pipefail

UDEV_FILE=/etc/udev/rules.d/90-can.rules
SYS_NET=/sys/class/net
LOCK_FILE=/run/lock/yam-can-setup.lock
ALL_TARGETS=(can_lead_l can_lead_r can_left can_right)
TARGET_NAMES=()
GUI=false
WORK_DIR=

usage() {
    echo "Usage: sudo bash $0 [--target can_lead_l,can_lead_r,can_left,can_right]"
    echo "Bind all four adapters by default; --target selects a subset in the given order."
    echo "Unplug the selected adapters first, then connect one per prompt."
}

fail() { echo "Error: $*" >&2; exit 1; }

prompt() {
    if $GUI; then printf '@prompt %s\n' "$*"; else printf '\n%s Press Enter to continue...\n' "$*"; fi
    IFS= read -r _answer || fail "Setup cancelled (input closed)."
}

get_serial() {
    # Consume all output (no grep -m1/SIGPIPE under pipefail).
    udevadm info -a -p "$SYS_NET/$1" | awk -F '"' '/ATTRS\{serial\}==/ && !found {print $2; found=1}'
}

scan_devices() {
    local iface serial
    for iface in "$SYS_NET"/can*; do
        [ -d "$iface" ] || continue
        serial=$(get_serial "${iface##*/}")
        [ -n "$serial" ] || fail "No USB serial for ${iface##*/}; check the adapter and retry."
        [[ "$serial" =~ ^[a-zA-Z0-9_.:-]+$ ]] || fail "Unsupported USB serial for ${iface##*/}."
        printf '%s %s\n' "$serial" "${iface##*/}"
    done
}

reload_rules() {
    udevadm control --reload-rules
    udevadm settle --timeout=10
}

install_rules() {
    # Stage on the same filesystem so replacing the rules is atomic.
    local staged
    staged=$(mktemp "${UDEV_FILE}.XXXXXX")
    install -m 0644 "$1" "$staged"
    mv -f -- "$staged" "$UDEV_FILE"
    reload_rules
}

main() {
    local target selected name serial iface count new_serial new_iface backup result
    TARGET_NAMES=("${ALL_TARGETS[@]}")
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --target)
                [ "$#" -ge 2 ] && [ -n "$2" ] || fail "--target requires a comma-separated list."
                selected=$2
                [[ "$selected" != ,* && "$selected" != *, && "$selected" != *,,* ]] || fail "Empty target."
                IFS=',' read -ra TARGET_NAMES <<< "$selected"
                selected=" "
                for target in "${TARGET_NAMES[@]}"; do
                    case "$target" in
                        can_lead_l|can_lead_r|can_left|can_right) ;;
                        *) fail "Invalid target '$target'. Valid targets: ${ALL_TARGETS[*]}" ;;
                    esac
                    [[ "$selected" != *" $target "* ]] || fail "Duplicate target '$target'."
                    selected+="$target "
                done
                shift 2 ;;
            --gui) GUI=true; shift ;;
            --help|-h) usage; return 0 ;;
            *) usage; fail "Unknown option '$1'." ;;
        esac
    done
    [ "$(uname -s)" = Linux ] || fail "YAM CAN setup requires a Linux robot host."
    [ "$(id -u)" = 0 ] || fail "Run with sudo: sudo bash $0"
    for name in udevadm ip modprobe flock; do
        command -v "$name" >/dev/null || fail "Install '$name' on the robot host, then retry."
    done
    # Also excludes a second CLI/GUI process on the same host.
    exec 9>"$LOCK_FILE"
    flock -n 9 || fail "Another CAN setup is running; finish or cancel it first."

    echo "=== YAM CAN adapter setup ==="
    echo "Rules file: $UDEV_FILE"
    echo "Binding order: ${TARGET_NAMES[*]}"
    prompt "Reset any robot session and unplug the selected adapters (${TARGET_NAMES[*]}). Leave other adapters connected."

    WORK_DIR=$(mktemp -d)
    trap 'rm -rf -- "$WORK_DIR"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    # Back up before the first write; cancellation never removes existing bindings.
    if [ -f "$UDEV_FILE" ]; then
        backup=$(mktemp "${UDEV_FILE}.backup.XXXXXX")
        cp -p -- "$UDEV_FILE" "$backup"
        echo "Original rules saved to $backup"
        cp -- "$UDEV_FILE" "$WORK_DIR/original"
    else
        touch "$WORK_DIR/original"
    fi
    cp -- "$WORK_DIR/original" "$WORK_DIR/auto"
    if ! grep -q 'CAN device auto-configuration' "$WORK_DIR/auto"; then
        # Register gs_usb adapters and bring CAN up automatically at 1 Mbit/s.
        printf '\n%s\n' \
            '# CAN device auto-configuration' \
            'ACTION=="add|change", SUBSYSTEM=="usb", ATTR{idVendor}=="2833", ATTR{idProduct}=="b010", RUN+="/sbin/modprobe gs_usb"' \
            'ACTION=="add|change", SUBSYSTEMS=="usb", ATTRS{idVendor}=="2833", ATTRS{idProduct}=="b010", RUN+="/bin/sh -c '\''echo 2833 b010 > /sys/bus/usb/drivers/gs_usb/new_id'\''"' \
            'SUBSYSTEM=="net", ENV{INTERFACE}=="can*", ATTRS{bInterfaceNumber}=="00", RUN+="/sbin/ip link set $name type can bitrate 1000000", RUN+="/sbin/ip link set $name up"' \
            >> "$WORK_DIR/auto"
        install_rules "$WORK_DIR/auto"
        echo "Installed CAN auto-configuration rules."
    fi
    modprobe gs_usb
    scan_devices > "$WORK_DIR/seen"
    # Preserve every rule except the selected NAME assignments. Commit this only
    # after all adapters are identified, so cancellation keeps the old bindings.
    selected=$(IFS=,; echo "${TARGET_NAMES[*]}")
    awk -v targets="$selected" '
        BEGIN {n=split(targets, names, ",")}
        { for (i=1; i<=n; i++) if ($0 ~ "NAME[[:space:]]*[:]?=[[:space:]]*\"" names[i] "\"") next; print }
    ' "$WORK_DIR/auto" > "$WORK_DIR/rules"

    for target in "${TARGET_NAMES[@]}"; do
        while true; do
            prompt "Plug in only the adapter for $target. Previously identified adapters can stay connected."
            udevadm settle --timeout=10
            scan_devices > "$WORK_DIR/current"
            count=0; new_serial=; new_iface=
            while read -r serial iface; do
                if ! awk -v serial="$serial" '$1 == serial {found=1} END {exit !found}' "$WORK_DIR/seen"; then
                    count=$((count + 1)); new_serial=$serial; new_iface=$iface
                fi
            done < "$WORK_DIR/current"
            if [ "$count" -ne 1 ]; then
                echo "Detected $count new adapters; connect exactly one for $target and retry."
                continue
            fi
            if grep -qF "ATTRS{serial}==\"$new_serial\"" "$WORK_DIR/rules"; then
                echo "Serial $new_serial already has an unselected binding. Unplug it and connect $target."
                continue
            fi
            printf 'SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="%s", NAME="%s"\n' \
                "$new_serial" "$target" >> "$WORK_DIR/rules"
            printf '%s %s\n' "$new_serial" "$new_iface" >> "$WORK_DIR/seen"
            echo "Identified $new_iface (serial $new_serial) as $target."
            break
        done
    done

    echo "Proposed rules:"
    cat "$WORK_DIR/rules"
    prompt "Install these bindings for ${TARGET_NAMES[*]}? Cancel to keep the existing bindings."
    install_rules "$WORK_DIR/rules"
    echo "Bindings installed. Replug the selected adapters to apply their names."
    prompt "Unplug ALL selected adapters and plug them back in."
    udevadm settle --timeout=10
    result=0
    for target in "${TARGET_NAMES[@]}"; do
        if ! ip -details link show "$target"; then
            echo "$target: not found. Replug its adapter and check the rules in $UDEV_FILE."
            result=1
        fi
    done
    [ "$result" = 0 ] || return "$result"
    echo "YAM CAN setup complete. Return to Collect and select the named buses."
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
