import requests
import time
import sys

def verify_system():
    print("Verifying System...")
    
    # 1. Check if Enhance Button is gone (by checking HTML content)
    try:
        response = requests.get("http://localhost:5000/")
        if response.status_code == 200:
            if "toggleEnhancement" in response.text or "btn-enhance" in response.text:
                 print("FAIL: 'Enhance' button or logic still present in HTML.")
            else:
                 print("PASS: 'Enhance' button removed from HTML.")
        else:
            print(f"FAIL: Could not fetch index page. Status: {response.status_code}")
    except Exception as e:
        print(f"FAIL: Error fetching index page: {e}")

    # 2. Check /status endpoint structure
    try:
        response = requests.get("http://localhost:5000/status")
        if response.status_code == 200:
            data = response.json()
            if "enhancement_enabled" in data:
                print("FAIL: 'enhancement_enabled' field still in /status response.")
            else:
                print("PASS: 'enhancement_enabled' removed from /status.")
        else:
            print(f"FAIL: Could not fetch status. Status: {response.status_code}")
    except Exception as e:
        print(f"FAIL: Error fetching status: {e}")

    # 3. Trigger Translation (Simulate Input)
    try:
        # First set language to Spanish to test translation
        requests.post("http://localhost:5000/set_language", json={"language": "Spanish"})
        
        # Simulate text
        print("Simulating input 'Hello'...")
        response = requests.post("http://localhost:5000/simulate_input", json={"text": "Hello"})
        if response.status_code == 200:
            print("PASS: Simulation triggered.")
            # We can't easily check the console output of app.py from here, 
            # but the user/agent can verify the logs show Riva/Offline usage.
        else:
             print(f"FAIL: Simulation failed. Status: {response.status_code}")
    except Exception as e:
        print(f"FAIL: Error triggering simulation: {e}")

if __name__ == "__main__":
    # Wait for app to start
    print("Waiting for app to start...")
    for i in range(10):
        try:
            requests.get("http://localhost:5000/")
            print("App is running!")
            break
        except:
            time.sleep(2)
            print(f"Waiting... ({i+1}/10)")
    else:
         print("Timeout: App did not start.")
         sys.exit(1)

    verify_system()
