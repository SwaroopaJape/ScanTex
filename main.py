import os

def main():
    print("Starting ScanTeX UI...")
    os.system("uv run streamlit run ui/app.py")

if __name__ == "__main__":
    main()
