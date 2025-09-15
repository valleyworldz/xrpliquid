#!/bin/bash
# Emergency Stop Script

cd /home/ubuntu/hypeliquidOG
python3 -c "
from critical_loop_control_system import get_emergency_stop_system
emergency = get_emergency_stop_system()
emergency.force_emergency_stop('Emergency script activated')
print('🚨 Emergency stop activated!')
"
