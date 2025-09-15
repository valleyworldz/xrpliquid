@echo off
set HL_ENCRYPTED_CREDS=Z0FBQUFBQm9hcnk4RFR0NHpWc1Q0b2xsYUwxT0E0Z2ZZUGVKM2NJSVlVeFlIWEYyZnBlc3Y5S2JrYnVWWHZ1aTc0Z25WOTRZOEtrTWV4N2pveGtPT29LTUktWjYyZ3N1QVdyM0U5YWUyeVIwSkpqX0FhRVVlUVVZZlRYLUQ0YkYtdFVKWXRQSDJCQ3VYXzZlUGo2RWtuMjdIc1NKa0dSSjFCZ09ybExTM2ttQmdrbDREUkEzcUM2TUpPWFV6S0Y1LWJhdndnSEdwWjRrNjJYMUNuZXVZZFIyeU1QWm5QUVRMcFMyTUs4N0N6VFptemRoQ0Y5bjFxMmwwUnZMakF2aHllZ3NINEx3ak1feA==
set HL_CRED_PASSWORD=HyperLiquidSecure2025!
set ENCRYPTED_PRIVATE_KEY=0xd0763c10747703e2b66e3de402a821924301d9d7b7847c334f26e92f06b3bcdf
set ENCRYPTED_ADDRESS=0x2FD162968bf87dFbF5e153E9b11ca64b0e73aB19

python -c "from core.utils.credential_handler import SecureCredentialHandler; handler = SecureCredentialHandler(); print('Testing...'); handler.initialize() and print('✅ Init OK') or print('❌ Init failed'); creds = handler.load_credentials(); creds and print('✅ Load OK') or print('❌ Load failed')" 