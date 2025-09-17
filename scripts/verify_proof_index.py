"""
Verify Proof Index - Check all URLs in PROOF_ARTIFACTS_SUMMARY.md return 200
"""

import requests
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
import time

def extract_urls_from_markdown(file_path: str) -> List[str]:
    """Extract all URLs from markdown file"""
    urls = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all markdown links [text](url)
    url_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    matches = re.findall(url_pattern, content)
    
    for text, url in matches:
        if url.startswith('http'):
            urls.append(url)
    
    return urls

def check_url_status(url: str, timeout: int = 10) -> Tuple[bool, int, str]:
    """Check if URL returns 200 status"""
    try:
        response = requests.get(url, timeout=timeout, allow_redirects=True)
        return True, response.status_code, response.reason
    except requests.exceptions.RequestException as e:
        return False, 0, str(e)

def verify_proof_index():
    """Verify all URLs in PROOF_ARTIFACTS_SUMMARY.md"""
    print("🔍 Verifying Proof Index URLs")
    print("=" * 50)
    
    # Extract URLs
    urls = extract_urls_from_markdown("PROOF_ARTIFACTS_SUMMARY.md")
    
    print(f"📊 Found {len(urls)} URLs to verify")
    
    results = []
    success_count = 0
    
    for i, url in enumerate(urls, 1):
        print(f"\n🔗 [{i}/{len(urls)}] Checking: {url}")
        
        success, status_code, reason = check_url_status(url)
        
        if success and status_code == 200:
            print(f"✅ Status: {status_code} - {reason}")
            success_count += 1
        else:
            print(f"❌ Status: {status_code} - {reason}")
        
        results.append({
            'url': url,
            'success': success,
            'status_code': status_code,
            'reason': reason
        })
        
        # Small delay to be respectful
        time.sleep(0.5)
    
    # Summary
    print(f"\n" + "=" * 50)
    print(f"📊 VERIFICATION SUMMARY:")
    print(f"  Total URLs: {len(urls)}")
    print(f"  Successful (200): {success_count}")
    print(f"  Failed: {len(urls) - success_count}")
    print(f"  Success Rate: {success_count/len(urls)*100:.1f}%")
    
    # Show failed URLs
    failed_urls = [r for r in results if not r['success'] or r['status_code'] != 200]
    if failed_urls:
        print(f"\n❌ FAILED URLS:")
        for result in failed_urls:
            print(f"  {result['url']} - {result['status_code']} {result['reason']}")
    
    # Save results
    with open('proof_index_verification_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: proof_index_verification_results.json")
    
    return success_count == len(urls)

if __name__ == "__main__":
    success = verify_proof_index()
    if success:
        print(f"\n🎉 ALL PROOF INDEX URLS VERIFIED SUCCESSFULLY!")
    else:
        print(f"\n⚠️ Some URLs failed verification - check results above")
