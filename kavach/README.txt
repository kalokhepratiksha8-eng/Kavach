KAVACH-AAYUSH demo prototype (offline, runs on localhost)
1) python -m venv venv  &&  venv\Scripts\activate      (Linux/Mac: source venv/bin/activate)
2) pip install -r requirements.txt
3) python main.py
4) Console: http://127.0.0.1:8000     Micro-app: http://127.0.0.1:8000/app   (open on phone via your PC's LAN IP if you want)
Demo flow: Micro-app -> take test as J-214 -> Console (RMO) -> click J-214 -> switch role to Commander -> Run rebalance -> Approve.
"Reset demo" button restores the starting data. Data is synthetic. Optional: pip install sqlcipher3-binary for real AES-256 DB encryption.
