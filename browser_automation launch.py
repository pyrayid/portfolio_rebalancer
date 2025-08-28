from playwright.sync_api import sync_playwright
import pandas as pd
import time


def read_trading_orders():
    """Read and process the trading orders from the CSV file."""
    df = pd.read_csv(
        "files/Portfolio_Rebalance_Hierarchical Risk Parity_Jun-23-2025.csv"
    )

    # Filter for rows with actual trades (buy or sell)
    buy_orders = df[df["shares_to_buy"] > 0][["Symbol", "shares_to_buy"]]
    sell_orders = df[df["shares_to_sell"] > 0][["Symbol", "shares_to_sell"]]

    return buy_orders, sell_orders


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

    with sync_playwright() as p:
        try:
            # Launch Chrome in a new window
            browser = p.chromium.launch(headless=False, args=["--start-maximized"])
            context = browser.new_context(viewport={"width": 1920, "height": 1080})
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
            if "browser" in locals() and browser:
                try:
                    print("Closing browser...")
                    browser.close()
                except Exception as e:
                    print(f"Error closing browser: {str(e)}")
            input("Press Enter to exit...")


if __name__ == "__main__":
    automate_fidelity()
