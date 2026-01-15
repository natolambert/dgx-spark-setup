#!/bin/bash
# oom_protection.sh - Memory safety utilities for DGX Spark unified memory
#
# DGX Spark has 128GB unified CPU/GPU memory. OOM can freeze the entire system.
#
# Usage:
#   source oom_protection.sh   # Load functions into current shell
#   ./oom_protection.sh check  # Check memory (standalone)
#   ./oom_protection.sh cleanup # Kill leftover GPU processes
#   ./oom_protection.sh watch PID # Watch and kill if memory low

set -e

# Kill leftover ray/vLLM processes that may be holding memory
cleanup_gpu_processes() {
    echo "Checking for leftover GPU processes..."
    local found=0

    if pgrep -f "ray::" > /dev/null; then
        echo "  Found ray processes"
        found=1
    fi
    if pgrep -f "vllm" > /dev/null; then
        echo "  Found vLLM processes"
        found=1
    fi

    if [ "$found" -eq 1 ]; then
        echo "Cleaning up..."
        pkill -9 -f "ray::" 2>/dev/null || true
        pkill -9 -f "vllm" 2>/dev/null || true
        pkill -9 -f "VLLM" 2>/dev/null || true
        ray stop --force 2>/dev/null || true
        echo "Waiting 5s for memory release..."
        sleep 5
    else
        echo "  No leftover processes found"
    fi
}

# Check available memory
check_memory() {
    local min_gb="${1:-60}"
    local free_kb=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    local free_gb=$((free_kb / 1024 / 1024))

    echo "Available memory: ${free_gb}GB (minimum: ${min_gb}GB)"

    if [ "$free_gb" -lt "$min_gb" ]; then
        echo "ERROR: Not enough memory!"
        echo "Run: $0 cleanup"
        return 1
    fi
    echo "OK: Memory check passed"
}

# Watch a process and kill it if memory gets too low
watch_memory() {
    local pid="$1"
    local threshold_gb="${2:-16}"
    local threshold_kb=$((threshold_gb * 1024 * 1024))

    echo "Watching PID $pid, will kill if <${threshold_gb}GB available"

    while kill -0 "$pid" 2>/dev/null; do
        local avail_kb=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
        if [ "$avail_kb" -lt "$threshold_kb" ]; then
            local avail_gb=$((avail_kb / 1024 / 1024))
            echo "Memory low (${avail_gb}GB). Killing PID $pid..."
            kill -TERM "$pid" 2>/dev/null || true
            sleep 5
            kill -KILL "$pid" 2>/dev/null || true
            echo "Process killed to prevent system freeze"
            return 1
        fi
        sleep 2
    done
    echo "Process $pid completed normally"
}

# Pre-flight check before training
preflight() {
    echo "=== DGX Spark Pre-flight Check ==="
    cleanup_gpu_processes
    check_memory "${1:-60}"
    echo ""
    echo "Ready to start training"
}

# Run if executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-help}" in
        cleanup)
            cleanup_gpu_processes
            ;;
        check)
            check_memory "${2:-60}"
            ;;
        watch)
            if [ -z "$2" ]; then
                echo "Usage: $0 watch PID [min_gb]"
                exit 1
            fi
            watch_memory "$2" "${3:-16}"
            ;;
        preflight)
            preflight "${2:-60}"
            ;;
        *)
            echo "DGX Spark OOM Protection Utilities"
            echo ""
            echo "Usage: $0 <command> [args]"
            echo ""
            echo "Commands:"
            echo "  cleanup          Kill leftover ray/vLLM processes"
            echo "  check [min_gb]   Check available memory (default: 60GB)"
            echo "  watch PID [gb]   Watch process, kill if memory < gb (default: 16GB)"
            echo "  preflight [gb]   Run cleanup + check before training"
            echo ""
            echo "Or source this file to use functions directly:"
            echo "  source $0"
            echo "  cleanup_gpu_processes"
            echo "  check_memory 80"
            ;;
    esac
fi
