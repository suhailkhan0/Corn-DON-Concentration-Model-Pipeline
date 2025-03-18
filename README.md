# Corn-DON-Concentration-Model-Pipeline

## Prediction Results Web App

This is a Flask-based web application that displays prediction results, download CSV files, and view residual distribution graphs. It includes logging for debugging and monitoring.

---

## **Project Structure**
```
project_directory/
│-- templates/
│   ├── index.html    # HTML template for home page
    ├── results.html    # HTML template for rendering results
│-- app.py            # Flask backend
│-- index.py          # modular backend code
│-- index.ipynb       # jupyter file with results
│-- requirements.txt  # Required Python dependencies
│-- app.log           # Log file for backend activities
│-- README.md         # Documentation
```

---

## **Setup and Installation**

### **1️⃣ Clone the Repository**
```sh
git clone <repository_url>
cd project_directory
```

### **2️⃣ Create a Virtual Environment (Recommended)**
```sh
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### **3️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

---

## **Running the Flask Application**

### **Start the Server**
```sh
python app.py
```
By default, the application runs at: **http://127.0.0.1:5000/**

---

## **Logging Setup**
This project uses Python's `logging` module to track runtime details and errors.
- **Log file:** `app.log`
- **Logging Levels:** INFO, ERROR

## **Frontend Features**
- **Table Toggling:** Show/hide extra rows dynamically.
- **CSV Download:** Save prediction results as a CSV file.
- **Graph Display:** Show a residual distribution graph.

---

## **Troubleshooting**
- **Application does not start?**
  - Ensure Flask is installed: `pip install flask`
  - Check Python version: `python --version`
- **JavaScript not working?**
  - Open browser console (`F12` → Console) to check for errors.
- **Logs not being recorded?**
  - Ensure `app.log` exists and has write permissions.

---

## **Future Improvements**
- Add database support for storing predictions.
- Implement user authentication.
- Improve UI with additional charts.

---

## **License**
This project is open-source and available for modification and distribution.

---

### **Developed By:**
🚀 Mohd Suhail | Data Scientist 

