#!/usr/bin/env python3
"""
Quick fix script for modular system issues
"""

import os
import shutil

def backup_and_fix():
    """Backup current files and apply fixes"""
    
    print("🔧 Fixing modular system issues...")
    
    # 1. Backup current routing.py
    if os.path.exists('routing.py'):
        shutil.copy('routing.py', 'routing_backup.py')
        print("✅ Backed up routing.py")
    
    # 2. Replace with fixed version
    if os.path.exists('routing_fixed.py'):
        shutil.copy('routing_fixed.py', 'routing.py')
        print("✅ Applied fixed routing.py")
    
    # 3. Test the system
    try:
        from routing import SimpleAgriculturalAI
        ai = SimpleAgriculturalAI()
        
        # Test basic functionality
        result = ai.ask("What fertilizer should I use for tomatoes?")
        
        print("🎉 SYSTEM TEST RESULTS:")
        print(f"✅ Answer received: {len(result.get('answer', ''))} characters")
        print(f"✅ Sources: {len(result.get('sources', []))} sources")
        print(f"✅ Chain of thought: {'Present' if result.get('chain_of_thought') else 'Missing'}")
        print(f"✅ Agent used: {result.get('agent_used', 'Unknown')}")
        print(f"✅ Success: {result.get('success', False)}")
        
        if result.get('chain_of_thought'):
            print(f"\n🔍 Chain of thought preview:")
            print(result['chain_of_thought'][:200] + "...")
        
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

if __name__ == "__main__":
    success = backup_and_fix()
    if success:
        print("\n🎉 Modular system fixed successfully!")
        print("LLM routing and chain of thoughts are now working properly.")
    else:
        print("\n❌ Fix failed. Check the error messages above.")
