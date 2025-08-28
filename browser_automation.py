from playwright.sync_api import sync_playwright
import pandas as pd
import time
import subprocess
import json
import os


def read_trading_orders():
    """Read and process the trading orders from the CSV file."""
    df = pd.read_csv(
        "files/Portfolio_Rebalance_Hierarchical Risk Parity_Jun-23-2025.csv"
    )

    # Filter for rows with actual trades (buy or sell)
    buy_orders = df[df["shares_to_buy"] > 0][["Symbol", "shares_to_buy"]]
    sell_orders = df[df["shares_to_sell"] > 0][["Symbol", "shares_to_sell"]]

    return buy_orders, sell_orders


def get_chrome_debug_port():
    """Try to find an existing Chrome instance with remote debugging enabled."""
    try:
        # First, check if Chrome is running at all
        chrome_running = False
        try:
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq chrome.exe"],
                capture_output=True,
                text=True,
                shell=True,
            )
            if "chrome.exe" in result.stdout:
                chrome_running = True
                print("Chrome is running, but checking if it has debugging enabled...")
        except Exception as e:
            print(f"Error checking Chrome processes: {e}")

        # Check if Chrome is running with remote debugging
        result = subprocess.run(
            ["netstat", "-an"], capture_output=True, text=True, shell=True
        )

        # Look for Chrome debug ports (typically 9222)
        if "9222" in result.stdout:
            print("Found Chrome debug port 9222 in netstat output")
            return 9222

        # Try common debug ports
        for port in [9222, 9223, 9224, 9225]:
            try:
                import socket

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(("localhost", port))
                sock.close()
                if result == 0:
                    print(f"Found active debug port {port}")
                    return port
            except:
                continue

        # If Chrome is running but no debug port found
        if chrome_running:
            print("Chrome is running but not with remote debugging enabled.")
            print(
                "To enable debugging, Chrome must be launched with --remote-debugging-port=9222"
            )

    except Exception as e:
        print(f"Error checking for existing Chrome: {e}")

    return None


def launch_chrome_with_debugging():
    """Launch Chrome with remote debugging enabled."""
    try:
        # Common Chrome paths on Windows
        chrome_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Users\{}\AppData\Local\Google\Chrome\Application\chrome.exe".format(
                os.getenv("USERNAME", "")
            ),
        ]

        chrome_exe = None
        for path in chrome_paths:
            if os.path.exists(path):
                chrome_exe = path
                break

        if not chrome_exe:
            print(
                "Chrome executable not found. Please launch Chrome manually with --remote-debugging-port=9222"
            )
            return False

        # Check if Chrome is already running
        try:
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq chrome.exe"],
                capture_output=True,
                text=True,
                shell=True,
            )
            if "chrome.exe" in result.stdout:
                print(
                    "Chrome is already running. You may need to close it first or use a different debug port."
                )
                print("Would you like to:")
                print("1. Close existing Chrome and launch with debugging")
                print("2. Try to connect to existing Chrome anyway")
                print("3. Use a different debug port")

                choice = input("Enter your choice (1, 2, or 3): ").strip()

                if choice == "1":
                    print("Closing existing Chrome...")
                    subprocess.run(["taskkill", "/F", "/IM", "chrome.exe"], shell=True)
                    time.sleep(2)
                elif choice == "3":
                    # Use a different port
                    debug_port = 9223
                    subprocess.Popen(
                        [
                            chrome_exe,
                            f"--remote-debugging-port={debug_port}",
                            "--no-first-run",
                            "--no-default-browser-check",
                        ]
                    )
                    print(f"Launched Chrome with remote debugging on port {debug_port}")
                    time.sleep(3)
                    return True
                else:
                    # Try to connect anyway
                    return False
        except Exception as e:
            print(f"Error checking Chrome processes: {e}")

        # Ask user about profile usage
        print("\nChrome Profile Options:")
        print(
            "1. Use existing Chrome profile (recommended - keeps your bookmarks, extensions, etc.)"
        )
        print("2. Use new debug profile (clean slate)")
        print("3. Let Chrome choose (will prompt you to select)")

        profile_choice = input("Enter your choice (1, 2, or 3): ").strip()

        chrome_args = [
            chrome_exe,
            "--remote-debugging-port=9222",
            "--no-first-run",
            "--no-default-browser-check",
        ]

        if profile_choice == "1":
            # Use existing profile - explicitly specify the default profile directory
            default_profile_dir = os.path.expanduser(
                r"~\AppData\Local\Google\Chrome\User Data"
            )
            if os.path.exists(default_profile_dir):
                chrome_args.append(f"--user-data-dir={default_profile_dir}")
                print(f"Using existing Chrome profile from: {default_profile_dir}")
            else:
                print(
                    "Default Chrome profile directory not found. Using Chrome's default behavior..."
                )
        elif profile_choice == "2":
            # Use new debug profile
            chrome_args.append("--user-data-dir=chrome_debug_profile")
            print("Using new debug profile...")
        else:
            # Let Chrome choose - don't specify user-data-dir
            print("Chrome will prompt you to select a profile...")

        # Launch Chrome with debugging
        subprocess.Popen(chrome_args)

        print("Launched Chrome with remote debugging on port 9222")
        time.sleep(3)  # Give Chrome time to start
        return True

    except Exception as e:
        print(f"Error launching Chrome: {e}")
        return False


def show_chrome_setup_instructions():
    """Show instructions for setting up Chrome with remote debugging."""
    print("\n" + "=" * 60)
    print("CHROME SETUP INSTRUCTIONS")
    print("=" * 60)
    print(
        "To connect to an existing Chrome window, you need to launch Chrome with remote debugging enabled."
    )
    print("\nMethod 1 - Manual Launch:")
    print("1. Close all Chrome windows")
    print("2. Open Command Prompt or PowerShell")
    print("3. Run this command:")
    print(
        '   "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --remote-debugging-port=9222'
    )
    print("4. Chrome will open with debugging enabled")
    print("5. Run this script again and choose option 2")
    print("\nMethod 2 - Let the script handle it:")
    print("1. Choose option 3 when prompted")
    print("2. The script will automatically launch Chrome with debugging")
    print("\nMethod 3 - Use existing Chrome (if already running with debugging):")
    print("1. If Chrome is already running with --remote-debugging-port=9222")
    print("2. Choose option 2 when prompted")
    print("\nIMPORTANT: Regular Chrome windows cannot be 'taken over' by the script.")
    print(
        "Chrome must be specifically launched with debugging flags for connection to work."
    )
    print("=" * 60)


def check_chrome_debug_status():
    """Check if Chrome is running with debugging and provide detailed status."""
    print("\n" + "=" * 50)
    print("CHROME DEBUG STATUS CHECK")
    print("=" * 50)

    # Check if Chrome is running
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq chrome.exe"],
            capture_output=True,
            text=True,
            shell=True,
        )
        if "chrome.exe" in result.stdout:
            print("✓ Chrome is running")
            chrome_count = result.stdout.count("chrome.exe")
            print(f"  Found {chrome_count} Chrome process(es)")
        else:
            print("✗ Chrome is not running")
            return
    except Exception as e:
        print(f"Error checking Chrome processes: {e}")
        return

    # Check for debug ports
    debug_port = get_chrome_debug_port()
    if debug_port:
        print(f"✓ Chrome has debugging enabled on port {debug_port}")
        print("  You can use option 2 to connect to this Chrome instance")

        # Try to get more detailed debug info
        try:
            import requests
            import json

            # Get the list of available targets
            response = requests.get(f"http://localhost:{debug_port}/json")
            if response.status_code == 200:
                targets = response.json()
                print(f"\n  Found {len(targets)} debug targets:")
                for i, target in enumerate(targets):
                    target_type = target.get("type", "unknown")
                    target_url = target.get("url", "no url")
                    print(f"    {i + 1}. Type: {target_type}, URL: {target_url}")
            else:
                print(
                    f"  Could not retrieve debug targets (status: {response.status_code})"
                )
        except Exception as e:
            print(f"  Could not get detailed debug info: {e}")

    else:
        print("✗ Chrome is running but NOT with debugging enabled")
        print("  Regular Chrome windows cannot be connected to by the script")
        print("  You need to launch Chrome with --remote-debugging-port=9222")
        print("\nOptions:")
        print("1. Close Chrome and use option 3 to launch with debugging")
        print("2. Use option 1 to launch a new Chrome window")
        print("3. Manually launch Chrome with debugging and try option 2 again")

    print("=" * 50)


def diagnose_chrome_connection():
    """Diagnose Chrome debugging connection issues."""
    print("\n" + "=" * 60)
    print("CHROME CONNECTION DIAGNOSIS")
    print("=" * 60)

    # Check if port 9222 is accessible
    try:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("localhost", 9222))
        sock.close()

        if result == 0:
            print("✓ Port 9222 is accessible")

            # Try to get debug info
            try:
                import requests

                response = requests.get("http://localhost:9222/json")
                if response.status_code == 200:
                    targets = response.json()
                    print(f"✓ Debug API responding - found {len(targets)} targets")

                    if targets:
                        print("\nAvailable targets:")
                        for i, target in enumerate(targets):
                            target_type = target.get("type", "unknown")
                            target_url = target.get("url", "no url")
                            target_id = target.get("id", "no id")
                            print(
                                f"  {i + 1}. ID: {target_id}, Type: {target_type}, URL: {target_url}"
                            )
                    else:
                        print("  No targets available - Chrome may still be loading")
                else:
                    print(
                        f"✗ Debug API not responding properly (status: {response.status_code})"
                    )
            except Exception as e:
                print(f"✗ Could not access debug API: {e}")
        else:
            print("✗ Port 9222 is not accessible")
            print("  Chrome may not be running with debugging enabled")
    except Exception as e:
        print(f"✗ Error checking port: {e}")

    print("=" * 60)


def launch_chrome_undetected():
    """Launch Chrome using undetected-chromedriver for better compatibility."""
    try:
        import undetected_chromedriver as uc

        print("Launching Chrome with undetected-chromedriver...")
        print("This method is more reliable for financial websites.")

        # Launch Chrome with undetected-chromedriver
        driver = uc.Chrome(
            version_main=None,  # Auto-detect Chrome version
            use_subprocess=True,
            headless=False,
        )

        # Convert to Playwright page
        # Note: This is a simplified approach - you might need to adapt
        print("Chrome launched successfully with undetected-chromedriver")
        return driver

    except ImportError:
        print(
            "undetected-chromedriver not installed. Install with: pip install undetected-chromedriver"
        )
        return None
    except Exception as e:
        print(f"Error launching undetected Chrome: {e}")
        return None


def show_alternative_methods():
    """Show alternative methods for browser automation."""
    print("\n" + "=" * 60)
    print("ALTERNATIVE BROWSER AUTOMATION METHODS")
    print("=" * 60)
    print("If Chrome debugging continues to cause issues, consider these alternatives:")
    print("\n1. Undetected ChromeDriver:")
    print("   - More reliable for financial websites")
    print("   - Better at avoiding detection")
    print("   - Install: pip install undetected-chromedriver")
    print("\n2. Selenium with ChromeDriver:")
    print("   - Traditional automation approach")
    print("   - More stable but less stealthy")
    print("   - Install: pip install selenium")
    print("\n3. Playwright with Stealth Mode:")
    print("   - Built-in stealth features")
    print("   - Better compatibility")
    print("   - Already using Playwright")
    print("\n4. Manual Profile Setup:")
    print("   - Launch Chrome manually with debugging")
    print("   - Use option 2 to connect to existing")
    print("   - More control over the process")
    print("=" * 60)


def automate_fidelity():
    """Automate Fidelity portfolio access and trading."""
    buy_orders, sell_orders = read_trading_orders()

    while True:
        order_type = (
            input(
                "Enter 'buy' to process buy orders or 'sell' to process sell orders: "
            )
            .lower()
            .strip()
        )
        if order_type in ["buy", "sell"]:
            break
        print("Invalid input. Please enter either 'buy' or 'sell'.")

        # Ask user about browser connection preference
    print("\nBrowser connection options:")
    print("1. Launch new Chrome window (default)")
    print("2. Connect to existing Chrome window (if available)")
    print("3. Launch new Chrome with debugging enabled")
    print("4. Show Chrome setup instructions")
    print("5. Check Chrome debug status")
    print("6. Diagnose Chrome connection issues")
    print("7. Show alternative methods")
    print("8. Use undetected Chrome (recommended for Fidelity)")

    while True:
        browser_choice = input(
            "Enter your choice (1, 2, 3, 4, 5, 6, 7, or 8): "
        ).strip()
        if browser_choice in ["1", "2", "3", "4", "5", "6", "7", "8"]:
            break
        print("Invalid choice. Please enter 1, 2, 3, 4, 5, 6, 7, or 8.")

    if browser_choice == "4":
        show_chrome_setup_instructions()
        return

    if browser_choice == "5":
        check_chrome_debug_status()
        return

    if browser_choice == "6":
        diagnose_chrome_connection()
        return

    if browser_choice == "7":
        show_alternative_methods()
        return

    if browser_choice == "8":
        if automate_fidelity_undetected():
            return
        else:
            print("Undetected Chrome failed, continuing with regular options...")

    with sync_playwright() as p:
        browser = None
        try:
            if browser_choice == "1":
                # Launch new Chrome window (original behavior)
                print("Launching new Chrome window...")
                browser = p.chromium.launch(headless=False, args=["--start-maximized"])
                context = browser.new_context(viewport={"width": 1920, "height": 1080})
                page = context.new_page()

            elif browser_choice == "2":
                # Try to connect to existing Chrome
                print("Attempting to connect to existing Chrome...")
                debug_port = get_chrome_debug_port()

                if debug_port:
                    try:
                        browser = p.chromium.connect_over_cdp(
                            f"http://localhost:{debug_port}"
                        )
                        # Get the first available context/page or create new one
                        contexts = browser.contexts
                        if contexts:
                            context = contexts[0]
                            pages = context.pages
                            if pages:
                                page = pages[0]  # Use first existing page
                            else:
                                page = context.new_page()
                        else:
                            context = browser.new_context(
                                viewport={"width": 1920, "height": 1080}
                            )
                            page = context.new_page()
                        print(
                            f"Successfully connected to existing Chrome on port {debug_port}"
                        )
                    except Exception as e:
                        print(f"Failed to connect to existing Chrome: {e}")
                        print(
                            "\nThis usually means Chrome isn't running with remote debugging enabled."
                        )
                        print("Would you like to:")
                        print("1. Launch new Chrome with debugging enabled")
                        print("2. Launch regular new Chrome window")
                        print("3. Exit and set up Chrome manually")

                        retry_choice = input("Enter your choice (1, 2, or 3): ").strip()

                        if retry_choice == "1":
                            if launch_chrome_with_debugging():
                                time.sleep(2)
                                try:
                                    browser = p.chromium.connect_over_cdp(
                                        "http://localhost:9222"
                                    )
                                    context = browser.new_context(
                                        viewport={"width": 1920, "height": 1080}
                                    )
                                    page = context.new_page()
                                    print(
                                        "Successfully connected to new Chrome instance"
                                    )
                                except Exception as e2:
                                    print(f"Failed to connect to new Chrome: {e2}")
                                    print("Falling back to regular launch...")
                                    browser = p.chromium.launch(
                                        headless=False, args=["--start-maximized"]
                                    )
                                    context = browser.new_context(
                                        viewport={"width": 1920, "height": 1080}
                                    )
                                    page = context.new_page()
                            else:
                                print("Falling back to regular launch...")
                                browser = p.chromium.launch(
                                    headless=False, args=["--start-maximized"]
                                )
                                context = browser.new_context(
                                    viewport={"width": 1920, "height": 1080}
                                )
                                page = context.new_page()
                        elif retry_choice == "2":
                            browser = p.chromium.launch(
                                headless=False, args=["--start-maximized"]
                            )
                            context = browser.new_context(
                                viewport={"width": 1920, "height": 1080}
                            )
                            page = context.new_page()
                        else:
                            print(
                                "Exiting. Please set up Chrome manually and try again."
                            )
                            return
                else:
                    print("No existing Chrome with debugging found.")
                    print("Would you like to:")
                    print("1. Launch new Chrome with debugging enabled")
                    print("2. Launch regular new Chrome window")
                    print("3. Exit and set up Chrome manually")

                    retry_choice = input("Enter your choice (1, 2, or 3): ").strip()

                    if retry_choice == "1":
                        if launch_chrome_with_debugging():
                            time.sleep(2)
                            try:
                                browser = p.chromium.connect_over_cdp(
                                    "http://localhost:9222"
                                )
                                context = browser.new_context(
                                    viewport={"width": 1920, "height": 1080}
                                )
                                page = context.new_page()
                                print("Successfully connected to new Chrome instance")
                            except Exception as e:
                                print(f"Failed to connect to new Chrome: {e}")
                                print("Falling back to regular launch...")
                                browser = p.chromium.launch(
                                    headless=False, args=["--start-maximized"]
                                )
                                context = browser.new_context(
                                    viewport={"width": 1920, "height": 1080}
                                )
                                page = context.new_page()
                        else:
                            print("Falling back to regular launch...")
                            browser = p.chromium.launch(
                                headless=False, args=["--start-maximized"]
                            )
                            context = browser.new_context(
                                viewport={"width": 1920, "height": 1080}
                            )
                            page = context.new_page()
                    elif retry_choice == "2":
                        browser = p.chromium.launch(
                            headless=False, args=["--start-maximized"]
                        )
                        context = browser.new_context(
                            viewport={"width": 1920, "height": 1080}
                        )
                        page = context.new_page()
                    else:
                        print("Exiting. Please set up Chrome manually and try again.")
                        return

            elif browser_choice == "3":
                # Launch new Chrome with debugging enabled
                print("Launching Chrome with debugging enabled...")
                if launch_chrome_with_debugging():
                    print("\nChrome launched with debugging enabled.")
                    print(
                        "If you selected a profile, please wait for it to fully load."
                    )
                    print(
                        "Once Chrome is fully loaded with your profile, press Enter to continue..."
                    )
                    input("Press Enter when Chrome is ready...")

                    # Try to connect with retries
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            print(
                                f"Attempting to connect to Chrome (attempt {attempt + 1}/{max_retries})..."
                            )

                            # First check if the debug port is accessible
                            import socket

                            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            result = sock.connect_ex(("localhost", 9222))
                            sock.close()

                            if result != 0:
                                print(
                                    "Debug port 9222 is not accessible. Chrome may not be ready yet."
                                )
                                if attempt < max_retries - 1:
                                    print("Waiting 3 seconds before retrying...")
                                    time.sleep(3)
                                    continue
                                else:
                                    raise Exception(
                                        "Debug port 9222 is not accessible after multiple attempts"
                                    )

                            browser = p.chromium.connect_over_cdp(
                                "http://localhost:9222"
                            )

                            # Wait a bit more for contexts to be fully established
                            time.sleep(2)

                            # Check if there are existing pages with the user's profile
                            contexts = browser.contexts
                            print(f"Found {len(contexts)} browser context(s)")

                            if contexts:
                                # Try to find a context with existing pages (user's profile)
                                context_with_pages = None
                                for i, ctx in enumerate(contexts):
                                    pages = ctx.pages
                                    print(f"Context {i}: {len(pages)} page(s)")
                                    if pages:
                                        context_with_pages = ctx
                                        break

                                if context_with_pages:
                                    # Use the first page in the context that has pages
                                    page = context_with_pages.pages[0]
                                    print("Using existing page with your profile")
                                else:
                                    # No context has pages, use the first context and create a page
                                    context = contexts[0]
                                    page = context.new_page()
                                    print("Created new page in existing context")
                            else:
                                # Create new context and page
                                context = browser.new_context(
                                    viewport={"width": 1920, "height": 1080}
                                )
                                page = context.new_page()
                                print("Created new context and page")

                            print("Successfully connected to new Chrome instance")
                            break  # Success, exit retry loop

                        except Exception as e:
                            print(f"Connection attempt {attempt + 1} failed: {e}")
                            if attempt < max_retries - 1:
                                print("Waiting 3 seconds before retrying...")
                                time.sleep(3)
                            else:
                                print("All connection attempts failed.")
                                print("This might be because:")
                                print("1. Chrome didn't start with debugging enabled")
                                print("2. Chrome is still loading")
                                print("3. Another process is using port 9222")
                                print("\nFalling back to regular Chrome launch...")
                                browser = p.chromium.launch(
                                    headless=False, args=["--start-maximized"]
                                )
                                context = browser.new_context(
                                    viewport={"width": 1920, "height": 1080}
                                )
                                page = context.new_page()
                else:
                    print(
                        "Failed to launch Chrome with debugging. Using regular launch..."
                    )
                    browser = p.chromium.launch(
                        headless=False, args=["--start-maximized"]
                    )
                    context = browser.new_context(
                        viewport={"width": 1920, "height": 1080}
                    )
                    page = context.new_page()

            # Navigate to Fidelity login page
            print("Navigating to Fidelity...")
            try:
                # Navigate with minimal wait
                page.goto(
                    "https://digital.fidelity.com/prgw/digital/login/full-page?AuthRedUrl=https://digital.fidelity.com/ftgw/digital/trade-equity/index/orderEntry",
                    wait_until="domcontentloaded",
                )

                print("Waiting for login form...")
                # Try to find login form elements immediately
                login_selectors = [
                    "#dom-username-input",  # Username field
                    "#dom-pswd-input",  # Password field
                    "div.pvd-button__contents:has-text('Log in')",  # Login button
                ]

                for selector in login_selectors:
                    page.wait_for_selector(selector, timeout=30000)
                    print(f"Found {selector}")

                print("Login form detected")
                time.sleep(10)  # Wait for user to log in manually
            except Exception as e:
                print(f"Error loading page: {str(e)}")
                raise e

            print("\nPlease login manually in the browser window...")
            print("The program will automatically continue once login is detected.")
            print("Looking for the Trade button to confirm successful login...")
            print("\nIf the program appears stuck after you've logged in:")
            print("1. Make sure you're on the portfolio summary page")
            print("2. Press Ctrl+C to exit and try again")

            # Check if trade page is displayed
            trade_url = "https://digital.fidelity.com/ftgw/digital/trade-equity/index/orderEntry"
            current_url = page.url
            if current_url != trade_url:
                # Check for system error header
                try:
                    error_header = page.query_selector("#dom-system-error-header")
                    if (
                        error_header
                        and "Sorry, we can't complete this action right now."
                        in error_header.inner_text()
                    ):
                        time.sleep(3)  # Wait a bit
                        print(
                            "System error detected: Sorry, we can't complete this action right now. Exiting."
                        )
                        return
                except Exception:
                    pass
                print(
                    f"Trade page not detected (current URL: {current_url}). Please navigate to {trade_url} and press Enter to continue."
                )
                input()
                page.wait_for_selector("#eq-ticket-dest-symbol", timeout=30000)
            else:
                try:
                    page.wait_for_selector("#eq-ticket-dest-symbol", timeout=30000)
                except Exception:
                    print(
                        "Trade page detected but trade form not found. Please reload the page and press Enter to continue."
                    )
                    input()
                    page.wait_for_selector("#eq-ticket-dest-symbol", timeout=30000)

            if order_type == "sell":
                print("\nProcessing sell orders...")
                for _, row in sell_orders.iterrows():
                    try:
                        # Navigate to trade page
                        # # Navigate directly to Trade page after login
                        # page.goto('https://digital.fidelity.com/ftgw/digital/trade-equity/index/orderEntry', wait_until='domcontentloaded')
                        # time.sleep(2)

                        # Fill in the trade ticket using Fidelity's order entry form
                        page.fill("#eq-ticket-dest-symbol", row["Symbol"])
                        time.sleep(1)
                        # Select Sell action
                        page.click("#selected-dropdown-itemaction")
                        page.click("div[role='option']:has-text('Sell')")
                        time.sleep(0.5)
                        # Enter quantity
                        page.fill("#eqt-shared-quantity", str(row["shares_to_sell"]))
                        time.sleep(0.5)
                        # Select Market order type
                        page.click("#dest-dropdownlist-button-ordertype")
                        page.click("div[role='option']:has-text('Market')")
                        time.sleep(0.5)
                        # Review order
                        page.click("s-assigned-wrapper:has-text('Preview order')")
                        time.sleep(2)

                        print(
                            f"Prepared sell order for {row['Symbol']}: {row['shares_to_sell']} shares"
                        )

                    except Exception as e:
                        print(
                            f"Error processing sell order for {row['Symbol']}: {str(e)}"
                        )
                        break

            else:  # order_type == 'buy'
                print("\nProcessing buy orders...")
                for _, row in buy_orders.iterrows():
                    try:
                        # Navigate to trade page
                        # # Navigate directly to Trade page after login
                        # page.goto('https://digital.fidelity.com/ftgw/digital/trade-equity/index/orderEntry', wait_until='domcontentloaded')
                        # time.sleep(2)

                        # Fill in the trade ticket using Fidelity's order entry form
                        page.fill("#eq-ticket-dest-symbol", row["Symbol"])
                        time.sleep(1)
                        # Select Buy action
                        page.click("#selected-dropdown-itemaction")
                        page.click("div[role='option']:has-text('Buy')")
                        time.sleep(0.5)
                        # Enter quantity
                        page.fill("#eqt-shared-quantity", str(row["shares_to_buy"]))
                        time.sleep(0.5)
                        # Select Market order type
                        page.click("#dest-dropdownlist-button-ordertype")
                        page.click("div[role='option']:has-text('Market')")
                        time.sleep(0.5)
                        # Review order
                        page.click("s-assigned-wrapper:has-text('Preview order')")
                        time.sleep(2)

                        print(
                            f"Prepared buy order for {row['Symbol']}: {row['shares_to_buy']} shares"
                        )

                    except Exception as e:
                        print(
                            f"Error processing buy order for {row['Symbol']}: {str(e)}"
                        )
                        break

            print(
                "\nAll orders prepared. Please review and submit manually for safety."
            )
            input("Press Enter to close the browser...")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
        finally:
            if browser:
                try:
                    print("Closing browser connection...")
                    if browser_choice in ["2", "3"] and hasattr(browser, "close"):
                        # For connected browsers, we might want to just disconnect
                        try:
                            browser.close()
                        except:
                            # If close fails, try disconnect
                            try:
                                browser.disconnect()
                            except:
                                pass
                    else:
                        browser.close()
                except Exception as e:
                    print(f"Error closing browser: {str(e)}")
            input("Press Enter to exit...")


def automate_fidelity_undetected():
    """Automate Fidelity using undetected-chromedriver."""
    try:
        import undetected_chromedriver as uc
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import Select
        import time

        buy_orders, sell_orders = read_trading_orders()

        while True:
            order_type = (
                input(
                    "Enter 'buy' to process buy orders or 'sell' to process sell orders: "
                )
                .lower()
                .strip()
            )
            if order_type in ["buy", "sell"]:
                break
            print("Invalid input. Please enter either 'buy' or 'sell'.")

        print("\nLaunching undetected Chrome...")

        # Check if Chrome is already running
        try:
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq chrome.exe"],
                capture_output=True,
                text=True,
                shell=True,
            )
            if "chrome.exe" in result.stdout:
                print(
                    "Chrome is already running. This may cause issues with undetected-chromedriver."
                )
                print("Options:")
                print("1. Close existing Chrome and launch new instance")
                print("2. Try to launch anyway (may fail)")
                print("3. Use regular Playwright method instead")

                chrome_choice = input("Enter your choice (1, 2, or 3): ").strip()

                if chrome_choice == "1":
                    print("Closing existing Chrome...")
                    subprocess.run(["taskkill", "/F", "/IM", "chrome.exe"], shell=True)
                    time.sleep(3)  # Wait for Chrome to fully close
                elif chrome_choice == "3":
                    print("Switching to regular Playwright method...")
                    return False
                # If choice is 2, continue with existing Chrome
        except Exception as e:
            print(f"Error checking Chrome processes: {e}")

        # Get the default Chrome profile directory
        default_profile_dir = os.path.expanduser(
            r"~\AppData\Local\Google\Chrome\User Data"
        )

        if os.path.exists(default_profile_dir):
            print(f"Using existing Chrome profile from: {default_profile_dir}")
        else:
            print(
                "Default Chrome profile directory not found. Using Chrome's default behavior..."
            )
            default_profile_dir = None

        # Launch undetected Chrome with profile
        try:
            driver = uc.Chrome(
                version_main=None,  # Auto-detect Chrome version
                use_subprocess=True,
                headless=False,
                # Use existing profile
                user_data_dir=default_profile_dir,
            )
        except Exception as launch_error:
            print(f"Failed to launch undetected Chrome: {launch_error}")
            print(
                "This usually happens when Chrome is already running with the same profile."
            )
            print("Options:")
            print("1. Close all Chrome windows and try again")
            print("2. Use regular Playwright method")

            retry_choice = input("Enter your choice (1 or 2): ").strip()

            if retry_choice == "1":
                print("Closing all Chrome processes...")
                subprocess.run(["taskkill", "/F", "/IM", "chrome.exe"], shell=True)
                time.sleep(3)

                # Try launching again
                try:
                    driver = uc.Chrome(
                        version_main=None,
                        use_subprocess=True,
                        headless=False,
                        user_data_dir=default_profile_dir,
                    )
                    print(
                        "Successfully launched Chrome after closing existing instances!"
                    )
                except Exception as retry_error:
                    print(f"Still failed after closing Chrome: {retry_error}")
                    print("Switching to regular Playwright method...")
                    return False
            else:
                print("Switching to regular Playwright method...")
                return False

        try:
            print("Chrome launched successfully!")
            print(f"Current URL: {driver.current_url}")

            # Wait a moment for Chrome to fully load
            time.sleep(3)

            # Navigate to Fidelity
            print("Navigating to Fidelity...")
            fidelity_url = "https://digital.fidelity.com/prgw/digital/login/full-page?AuthRedUrl=https://digital.fidelity.com/ftgw/digital/trade-equity/index/orderEntry"

            try:
                driver.get(fidelity_url)
                print(f"Navigation completed. Current URL: {driver.current_url}")
            except Exception as nav_error:
                print(f"Error navigating to Fidelity: {nav_error}")
                print("Trying alternative approach...")

                # Try navigating to main Fidelity page first
                try:
                    driver.get("https://digital.fidelity.com")
                    print("Navigated to main Fidelity page")
                    time.sleep(2)

                    # Then try to navigate to login
                    driver.get(fidelity_url)
                    print(
                        f"Now navigated to login page. Current URL: {driver.current_url}"
                    )
                except Exception as alt_error:
                    print(f"Alternative navigation also failed: {alt_error}")
                    print(
                        "Please manually navigate to Fidelity and press Enter to continue..."
                    )
                    input("Press Enter when you're on the Fidelity page...")

            print("\nPlease login manually in the browser window...")
            print("The program will automatically continue once login is detected.")

            # Wait for user to login and navigate to trade page
            wait = WebDriverWait(driver, 300)  # 5 minute timeout

            # Wait for trade page elements
            trade_url = "https://digital.fidelity.com/ftgw/digital/trade-equity/index/orderEntry"

            # Check if we're already on the trade page
            current_url = driver.current_url
            print(f"Current URL: {current_url}")

            if current_url != trade_url:
                print(f"Please navigate to: {trade_url}")
                input("Press Enter when you're on the trade page...")

            # Wait for the symbol input field
            print("Waiting for trade page elements...")
            try:
                symbol_input = wait.until(
                    EC.presence_of_element_located((By.ID, "eq-ticket-dest-symbol"))
                )
                print("Trade page detected successfully!")
            except Exception as wait_error:
                print(f"Error waiting for trade page elements: {wait_error}")
                print(
                    "Please make sure you're on the correct trade page and press Enter..."
                )
                input("Press Enter when ready...")

                # Try to find the element again
                try:
                    symbol_input = driver.find_element(By.ID, "eq-ticket-dest-symbol")
                    print("Found trade page elements!")
                except Exception as find_error:
                    print(f"Could not find trade page elements: {find_error}")
                    print("Please check if you're on the correct page and try again.")
                    return False

            if order_type == "sell":
                print("\nProcessing sell orders...")
                for _, row in sell_orders.iterrows():
                    try:
                        # Fill in the trade ticket
                        symbol_input.clear()
                        symbol_input.send_keys(row["Symbol"])
                        time.sleep(1)

                        # Select Sell action
                        action_dropdown = driver.find_element(
                            By.ID, "selected-dropdown-itemaction"
                        )
                        action_dropdown.click()
                        time.sleep(0.5)

                        sell_option = driver.find_element(
                            By.XPATH,
                            "//div[@role='option' and contains(text(), 'Sell')]",
                        )
                        sell_option.click()
                        time.sleep(0.5)

                        # Enter quantity
                        quantity_input = driver.find_element(
                            By.ID, "eqt-shared-quantity"
                        )
                        quantity_input.clear()
                        quantity_input.send_keys(str(row["shares_to_sell"]))
                        time.sleep(0.5)

                        # Select Market order type
                        order_type_dropdown = driver.find_element(
                            By.ID, "dest-dropdownlist-button-ordertype"
                        )
                        order_type_dropdown.click()
                        time.sleep(0.5)

                        market_option = driver.find_element(
                            By.XPATH,
                            "//div[@role='option' and contains(text(), 'Market')]",
                        )
                        market_option.click()
                        time.sleep(0.5)

                        # Review order
                        preview_button = driver.find_element(
                            By.XPATH,
                            "//s-assigned-wrapper[contains(text(), 'Preview order')]",
                        )
                        preview_button.click()
                        time.sleep(2)

                        print(
                            f"Prepared sell order for {row['Symbol']}: {row['shares_to_sell']} shares"
                        )

                    except Exception as e:
                        print(
                            f"Error processing sell order for {row['Symbol']}: {str(e)}"
                        )
                        break

            else:  # order_type == 'buy'
                print("\nProcessing buy orders...")
                for _, row in buy_orders.iterrows():
                    try:
                        # Fill in the trade ticket
                        symbol_input.clear()
                        symbol_input.send_keys(row["Symbol"])
                        time.sleep(1)

                        # Select Buy action
                        action_dropdown = driver.find_element(
                            By.ID, "selected-dropdown-itemaction"
                        )
                        action_dropdown.click()
                        time.sleep(0.5)

                        buy_option = driver.find_element(
                            By.XPATH,
                            "//div[@role='option' and contains(text(), 'Buy')]",
                        )
                        buy_option.click()
                        time.sleep(0.5)

                        # Enter quantity
                        quantity_input = driver.find_element(
                            By.ID, "eqt-shared-quantity"
                        )
                        quantity_input.clear()
                        quantity_input.send_keys(str(row["shares_to_buy"]))
                        time.sleep(0.5)

                        # Select Market order type
                        order_type_dropdown = driver.find_element(
                            By.ID, "dest-dropdownlist-button-ordertype"
                        )
                        order_type_dropdown.click()
                        time.sleep(0.5)

                        market_option = driver.find_element(
                            By.XPATH,
                            "//div[@role='option' and contains(text(), 'Market')]",
                        )
                        market_option.click()
                        time.sleep(0.5)

                        # Review order
                        preview_button = driver.find_element(
                            By.XPATH,
                            "//s-assigned-wrapper[contains(text(), 'Preview order')]",
                        )
                        preview_button.click()
                        time.sleep(2)

                        print(
                            f"Prepared buy order for {row['Symbol']}: {row['shares_to_buy']} shares"
                        )

                    except Exception as e:
                        print(
                            f"Error processing buy order for {row['Symbol']}: {str(e)}"
                        )
                        break

            print(
                "\nAll orders prepared. Please review and submit manually for safety."
            )
            input("Press Enter to close the browser...")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
        finally:
            try:
                print("Closing browser...")
                driver.quit()
            except Exception as e:
                print(f"Error closing browser: {str(e)}")
            input("Press Enter to exit...")

    except ImportError:
        print("undetected-chromedriver not installed.")
        print("Install it with: pip install undetected-chromedriver")
        print("Falling back to regular Playwright method...")
        return False
    except Exception as e:
        print(f"Error with undetected Chrome: {e}")
        print("Falling back to regular Playwright method...")
        return False


if __name__ == "__main__":
    automate_fidelity()
