#!/bin/bash
# This line tells the system to use the Bash shell to run the script

# Define a function named get_cpu_usage
get_cpu_usage() {
    # Read the first line of /proc/stat (which contains CPU statistics)
    # and store it in an array called PREV
    PREV=($(head -n1 /proc/stat))

    # Wait 1 second to get a time difference for usage calculation
    sleep 1

    # Read the first line of /proc/stat again after 1 second
    # and store it in an array called CURR
    CURR=($(head -n1 /proc/stat))

    # Extract the 'idle' CPU time (field 5) from both snapshots
    PREV_IDLE=${PREV[4]}
    CURR_IDLE=${CURR[4]}

    # Initialize total time counters for both snapshots
    PREV_TOTAL=0
    CURR_TOTAL=0

    # Loop through all values (excluding the first element, which is "cpu")
    # and calculate the total time from the first snapshot
    for value in "${PREV[@]:1}"; do
        PREV_TOTAL=$((PREV_TOTAL + value))
    done

    # Do the same for the second snapshot
    for value in "${CURR[@]:1}"; do
        CURR_TOTAL=$((CURR_TOTAL + value))
    done

    # Calculate the difference in idle time between the two snapshots
    DIFF_IDLE=$((CURR_IDLE - PREV_IDLE))

    # Calculate the difference in total time between the two snapshots
    DIFF_TOTAL=$((CURR_TOTAL - PREV_TOTAL))

    # Calculate CPU usage as a percentage:
    # (Total time - Idle time) / Total time * 100
    DIFF_USAGE=$((100 * (DIFF_TOTAL - DIFF_IDLE) / DIFF_TOTAL))

    # Print the CPU usage with a percent symbol
    echo "$DIFF_USAGE%"
}

# Print the static header and labels once
echo "Resources at Use"
echo "CPU Usage:"
echo "RAM Usage:"
echo "Available RAM:"

# Start an infinite loop to update system stats every 2 seconds
while true; do
    # Move the cursor up 3 lines to overwrite the previous readings
    tput cuu 3

    # CPU usage label + value
    echo -ne "CPU Usage:      "
    get_cpu_usage

    # RAM Usage label + value (formatted with one decimal percent)
    echo -ne "RAM Usage:      "
    free -m | awk '/^Mem:/ {
        used = $2 - $7;
        used_pct = 100 * used / $2;
        printf "%dMB / %dMB (%.1f%%)\n", used, $2, used_pct;
    }'

    # Available RAM label + value (formatted with one decimal percent)
    echo -ne "Available RAM:  "
    free -m | awk '/^Mem:/ {
        available = $7;
        available_pct = 100 * available / $2;
        printf "%dMB (%.1f%%)\n", available, available_pct;
    }'
    sleep 2
done
