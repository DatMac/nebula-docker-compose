import urllib.request
import json
import time
import argparse
import concurrent.futures

# --- Configuration ---
DEFAULT_URL = "http://inference-api:8000/predict"
DEFAULT_REQUESTS = 1000
DEFAULT_CONCURRENCY = 50
DEFAULT_TIMEOUT = 10

# --- NEW: Define the range for sequential Customer IDs ---
CUST_ID_MIN = 1
CUST_ID_MAX = 1_000_000

# --- MODIFIED: The make_request function now accepts a cust_id ---
def make_request(url, timeout, cust_id):
    """
    Performs a single synchronous POST request with a specific customer ID.
    Returns a tuple of ('success' or 'error', latency_or_error_message).
    """
    payload = {"cust_id": cust_id}
    # Data must be encoded to bytes for urllib
    encoded_payload = json.dumps(payload).encode('utf-8')
    
    request = urllib.request.Request(
        url,
        data=encoded_payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )

    start_time = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            # Check for successful status code
            if 200 <= response.status < 300:
                latency = time.monotonic() - start_time
                return ('success', latency)
            else:
                latency = time.monotonic() - start_time
                return ('error', f"HTTPError: {response.status} for {cust_id} after {latency:.2f}s")
    except urllib.error.HTTPError as e:
        latency = time.monotonic() - start_time
        return ('error', f"HTTPError: {e.code} for {cust_id} after {latency:.2f}s")
    except Exception as e: # Catches timeouts, connection errors etc.
        latency = time.monotonic() - start_time
        return ('error', f"{e.__class__.__name__} for {cust_id} after {latency:.2f}s")


def print_report(total_time, results):
    """Prints a detailed report of the stress test results. (No changes here)"""
    successes = [latency for status, latency in results if status == 'success']
    errors = [error for status, error in results if status == 'error']
    
    total_requests = len(results)
    num_successes = len(successes)
    num_errors = len(errors)
    
    print("\n\n--- Stress Test Report ---")
    print(f"Total time:       {total_time:.2f} seconds")
    print(f"Total requests:   {total_requests}")
    print(f"Successful:       {num_successes}")
    print(f"Failed:           {num_errors}")
    
    if total_time > 0:
        rps = num_successes / total_time
        print(f"Requests Per Sec: {rps:.2f} (RPS)")
    
    if total_requests > 0:
        error_rate = (num_errors / total_requests) * 100
        print(f"Error Rate:       {error_rate:.2f}%")

    if num_successes > 0:
        latencies = sorted(successes)
        
        print("\n--- Latency (seconds) ---")
        print(f"Average:          {sum(latencies) / num_successes:.4f}")
        print(f"Min:              {latencies[0]:.4f}")
        print(f"Max:              {latencies[-1]:.4f}")
        
        print("\n--- Latency Percentiles (seconds) ---")
        p50_index = int(num_successes * 0.50)
        p90_index = int(num_successes * 0.90)
        p95_index = int(num_successes * 0.95)
        p99_index = int(num_successes * 0.99)
        print(f"p50 (Median):     {latencies[p50_index]:.4f}")
        print(f"p90:              {latencies[p90_index]:.4f}")
        print(f"p95:              {latencies[p95_index]:.4f}")
        print(f"p99:              {latencies[p99_index]:.4f}")

    if num_errors > 0:
        print("\n--- Error Summary ---")
        error_counts = {}
        for e in errors:
            error_counts[e] = error_counts.get(e, 0) + 1
        sorted_errors = sorted(error_counts.items(), key=lambda item: item[1], reverse=True)
        for error, count in sorted_errors[:5]:
            print(f"- [{count} times] {error}")
        if len(sorted_errors) > 5:
            print("... and more.")

def main():
    """Main function to set up and run the stress test."""
    parser = argparse.ArgumentParser(description="Native Python API Stress Tester with Sequential IDs")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="API endpoint URL")
    parser.add_argument("-n", "--requests", type=int, default=DEFAULT_REQUESTS, help="Total number of requests")
    parser.add_argument("-c", "--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Number of concurrent requests (threads)")
    parser.add_argument("-t", "--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request timeout in seconds")
    args = parser.parse_args()

    print(f"Starting stress test with {args.concurrency} concurrent workers for {args.requests} total requests to {args.url}...\n")
    
    results = []
    start_time = time.monotonic()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        # --- MODIFIED: Generate and submit tasks with sequential cust_ids ---
        futures = []
        cust_id_range_size = CUST_ID_MAX - CUST_ID_MIN + 1
        
        for i in range(args.requests):
            # Calculate the customer number, looping if necessary
            customer_number = (i % cust_id_range_size) + CUST_ID_MIN
            
            # Format the customer ID string: CUST-00000001, etc.
            cust_id = f"CUST-{customer_number:08}"
            
            # Submit the task with the generated cust_id
            futures.append(executor.submit(make_request, args.url, args.timeout, cust_id))
        
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                results.append(('error', str(exc)))
            
            completed_count += 1
            print(f"\rProgress: {completed_count}/{args.requests} requests completed", end="")

    total_time = time.monotonic() - start_time
    
    print_report(total_time, results)


if __name__ == "__main__":
    main()
