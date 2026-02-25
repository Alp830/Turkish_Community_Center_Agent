#!/usr/bin/env python
"""Helper script to set up Google Sheets authentication for the Church Finder Agent."""

import json
import os
import sys
from pathlib import Path


def setup_google_auth():
    """Interactive setup for Google Sheets authentication."""
    print("=" * 60)
    print("Google Sheets Authentication Setup")
    print("=" * 60)
    
    print("\nThis script will help you set up Google Sheets integration.")
    print("\nYou have two options:")
    print("1. Service Account (Recommended for automated tasks)")
    print("2. OAuth 2.0 (For manual/personal use)")
    
    choice = input("\nWhich method would you like to use? (1 or 2): ").strip()
    
    if choice == "1":
        setup_service_account()
    elif choice == "2":
        setup_oauth()
    else:
        print("Invalid choice. Please run this script again and choose 1 or 2.")
        return False
    
    return True


def setup_service_account():
    """Guide user through service account setup."""
    print("\n" + "=" * 60)
    print("Service Account Setup")
    print("=" * 60)
    
    print("\nFollow these steps to create a service account:")
    print("1. Go to https://console.cloud.google.com/")
    print("2. Create a new project (or select existing)")
    print("3. Go to APIs & Services > Library")
    print("4. Search for 'Google Sheets API' and enable it")
    print("5. Go to APIs & Services > Credentials")
    print("6. Click 'Create Credentials' > 'Service Account'")
    print("7. Fill in details and click 'Create and Continue'")
    print("8. In the Keys tab, click 'Add Key' > 'Create new key'")
    print("9. Choose JSON and click 'Create'")
    print("10. Save the JSON file")
    
    creds_path = input("\nEnter the path to your credentials.json file: ").strip()
    creds_path = Path(creds_path).expanduser()
    
    if not creds_path.exists():
        print(f"Error: File not found at {creds_path}")
        return False
    
    # Verify it's valid JSON
    try:
        with open(creds_path) as f:
            creds_data = json.load(f)
        
        if "client_email" not in creds_data:
            print("Error: credentials.json missing 'client_email' field")
            return False
        
        service_email = creds_data["client_email"]
        print(f"\nService account email: {service_email}")
        
    except json.JSONDecodeError:
        print("Error: Invalid JSON in credentials file")
        return False
    
    # Copy to project root
    dest_path = Path(__file__).parent / "credentials.json"
    print(f"\nCopying credentials.json to {dest_path}")
    dest_path.write_text(creds_path.read_text())
    
    print("\n[OK] Service account credentials saved!")
    print(f"\nNow, go to https://sheets.google.com and:")
    print("1. Create a new Google Sheet (or use existing one)")
    print(f"2. Share it with: {service_email}")
    print("3. Give it 'Editor' permissions")
    print("\nYour agent can now use Google Sheets!")
    
    return True


def setup_oauth():
    """Guide user through OAuth setup."""
    print("\n" + "=" * 60)
    print("OAuth 2.0 Setup")
    print("=" * 60)
    
    print("\nFollow these steps to create OAuth credentials:")
    print("1. Go to https://console.cloud.google.com/")
    print("2. Create a new project (or select existing)")
    print("3. Go to APIs & Services > Library")
    print("4. Search for 'Google Sheets API' and enable it")
    print("5. Go to APIs & Services > Credentials")
    print("6. Click 'Create Credentials' > 'OAuth 2.0 Client ID'")
    print("7. Choose 'Desktop application'")
    print("8. Click 'Create' and download the JSON")
    
    creds_path = input("\nEnter the path to your OAuth credentials JSON file: ").strip()
    creds_path = Path(creds_path).expanduser()
    
    if not creds_path.exists():
        print(f"Error: File not found at {creds_path}")
        return False
    
    # Verify it's valid JSON
    try:
        with open(creds_path) as f:
            json.load(f)
    except json.JSONDecodeError:
        print("Error: Invalid JSON in credentials file")
        return False
    
    # Copy to project root
    dest_path = Path(__file__).parent / "oauth_credentials.json"
    print(f"\nCopying OAuth credentials to {dest_path}")
    dest_path.write_text(creds_path.read_text())
    
    print("\n[OK] OAuth credentials saved!")
    print("\nOn first run, your agent will prompt you to authorize access.")
    print("Your agent can now use Google Sheets!")
    
    return True


def verify_setup():
    """Verify the authentication setup."""
    print("\n" + "=" * 60)
    print("Verifying Setup")
    print("=" * 60)
    
    creds_path = Path(__file__).parent / "credentials.json"
    oauth_path = Path(__file__).parent / "oauth_credentials.json"
    
    if creds_path.exists():
        print("[OK] Service account credentials found")
        return True
    elif oauth_path.exists():
        print("[OK] OAuth credentials found")
        return True
    else:
        print("[X] No credentials found")
        print("Please run this setup script to configure authentication")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify_setup()
    else:
        if setup_google_auth():
            print("\n[OK] Setup complete!")
            print("\nNext steps:")
            print("1. If using Service Account: Share a Google Sheet with the service account email")
            print("2. Run your agent to start finding churches and writing to the spreadsheet")
        else:
            print("\n[X] Setup failed. Please try again or refer to GOOGLE_SHEETS_SETUP.md")
