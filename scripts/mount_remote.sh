#!/bin/bash
# Copyright 2026 Zachary Olkin. All rights reserved.

# Script to mount remote robot_rl directory using SSHFS
# This allows you to access remote training logs as if they were local

set -e

# Configuration
#REMOTE_HOST="131.215.101.160" #"131.215.101.40" #"talos.caltech.edu"
REMOTE_HOST="talos.caltech.edu"
# REMOTE_HOST="10.42.0.1"
# REMOTE_PATH="/home/unitree/robot_rl/transfer/obelisk/ctrl_logs"
# REMOTE_USER="unitree"
#REMOTE_USER="amy"
REMOTE_USER="zolkin"
#REMOTE_HOST="vulcan.amberlab.caltech.edu"
#REMOTE_PATH="/home/amy/gitrepo/robot_rl/logs"
REMOTE_PATH="/home/zolkin/robot_rl/logs"
#LOCAL_MOUNT_POINT="amy_remote_mount"
#LOCAL_MOUNT_POINT="vulcan_remote_mount"
LOCAL_MOUNT_POINT="talos_remote_mount"
# LOCAL_MOUNT_POINT="vulcan_mount"
# LOCAL_MOUNT_POINT="g1_mount"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [mount|unmount|status]"
    echo ""
    echo "Commands:"
    echo "  mount     Mount the remote directory (default)"
    echo "  unmount   Unmount the remote directory"
    echo "  status    Check if the directory is mounted"
    echo ""
    echo "The remote directory will be mounted at: $LOCAL_MOUNT_POINT"
    echo "Remote path: $REMOTE_HOST:$REMOTE_PATH"
}

# Check if SSHFS is installed
check_sshfs() {
    if ! command -v sshfs &> /dev/null; then
        print_error "SSHFS is not installed. Please install it first:"
        echo "  Ubuntu/Debian: sudo apt install sshfs"
        echo "  CentOS/RHEL: sudo yum install fuse-sshfs"
        echo "  macOS: brew install sshfs"
        exit 1
    fi
}

# Check if mount point exists and is mounted
check_mount_status() {
    if mountpoint -q "$LOCAL_MOUNT_POINT" 2>/dev/null; then
        print_success "Remote directory is mounted at $LOCAL_MOUNT_POINT"
        return 0
    else
        print_info "Remote directory is not mounted"
        return 1
    fi
}

# Mount the remote directory
mount_remote() {
    print_info "Mounting remote directory..."
    
    # Create mount point if it doesn't exist
    if [[ ! -d "$LOCAL_MOUNT_POINT" ]]; then
        print_info "Creating mount point: $LOCAL_MOUNT_POINT"
        mkdir -p "$LOCAL_MOUNT_POINT"
    fi
    
    # Check if already mounted
    if check_mount_status; then
        print_warning "Directory is already mounted"
        return 0
    fi
    
    # Test SSH connection
    print_info "Testing SSH connection..."
    if ! ssh -o ConnectTimeout=10 "$REMOTE_USER@$REMOTE_HOST" "echo 'SSH connection successful'" 2>/dev/null; then
        print_error "Failed to connect to $REMOTE_HOST"
        print_info "Please ensure you have SSH access configured."
        exit 1
    fi
    
    # Mount using SSHFS
    print_info "Mounting $REMOTE_HOST:$REMOTE_PATH to $LOCAL_MOUNT_POINT"
    sshfs "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" "$LOCAL_MOUNT_POINT" -o follow_symlinks,default_permissions
    
    if check_mount_status; then
        print_success "Remote directory mounted successfully!"
        print_info "You can now access remote files at: $LOCAL_MOUNT_POINT"
        
        # List top-level contents of the mounted directory
        print_info "Top-level contents of $LOCAL_MOUNT_POINT:"
        ls -1 "$LOCAL_MOUNT_POINT" | sed 's/^/  /'

        # Recursively search and list any directories containing training logs or policies
        print_info "Searching for available training logs and policy directories..."
        find "$LOCAL_MOUNT_POINT" -type d \( -iname "*log*" -o -iname "*policy*" \) -maxdepth 3 | sed 's/^/  /'
    else
        print_error "Failed to mount remote directory"
        exit 1
    fi

}

# Unmount the remote directory
unmount_remote() {
    print_info "Unmounting remote directory..."
    
    if check_mount_status; then
        fusermount -u "$LOCAL_MOUNT_POINT" 2>/dev/null || umount "$LOCAL_MOUNT_POINT" 2>/dev/null
        print_success "Remote directory unmounted successfully!"
    else
        print_warning "Directory is not mounted"
    fi
}

# Main script logic
main() {
    check_sshfs
    
    case "${1:-mount}" in
        "mount")
            mount_remote
            ;;
        "unmount")
            unmount_remote
            ;;
        "status")
            check_mount_status
            ;;
        "-h"|"--help")
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 