# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 red1239109-cmd
#!/usr/bin/env python3
# ==============================================================================
# File: check_env.py
# Description: Pre-flight checklist for GDR System
# ==============================================================================
import sys
import os
import socket
import importlib.util
from pathlib import Path

def check_dependencies():
    print("📦 [1/5] Checking Dependencies...")
    required = ['numpy', 'pandas', 'scipy', 'flask', 'psutil']
    missing = []
    for pkg in required:
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
            print(f"   ❌ Missing: {pkg}")
        else:
            print(f"   ✅ Found: {pkg}")
    
    if missing:
        print(f"🚨 Please run: pip install {' '.join(missing)}")
        return False
    return True

def check_permissions():
    print("\nkT [2/5] Checking Directories...")
    runs_dir = Path("runs")
    try:
        runs_dir.mkdir(exist_ok=True)
        test_file = runs_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        print(f"   ✅ Write access to '{runs_dir}' confirmed.")
        return True
    except Exception as e:
        print(f"   ❌ Permission Denied: {e}")
        return False

def check_port(port=8080):
    print(f"\n🌐 [3/5] Checking Port {port}...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('localhost', port)) == 0:
            print(f"   ❌ Port {port} is ALREADY IN USE.")
            return False
        else:
            print(f"   ✅ Port {port} is available.")
            return True

def main():
    print("=== 🛡️ GDR Pre-flight Check ===\n")
    checks = [
        check_dependencies(),
        check_permissions(),
        check_port()
    ]
    
    if all(checks):
        print("\n✅ All systems GO. Ready to launch 'gdr_v2_1.py'.")
        sys.exit(0)
    else:
        print("\n🛑 System checks FAILED. Fix issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
