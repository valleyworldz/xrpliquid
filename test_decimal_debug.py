from decimal import Decimal
import traceback

try:
    value = 123.456
    decimal_value = Decimal(str(value))
    print('Decimal value:', decimal_value)
    
    pattern = '0.' + '0' * 8
    print('Pattern:', pattern)
    
    quantize_pattern = Decimal(pattern)
    print('Quantize pattern:', quantize_pattern)
    
    result = decimal_value.quantize(quantize_pattern)
    print('Success:', result)
    
except Exception as e:
    print('Error:', e)
    traceback.print_exc()
