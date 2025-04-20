import sqlite3

conn = sqlite3.connect('predictions.db')
c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    actual_value FLOAT,
    predicted_value FLOAT
)
''')

def log_prediction(actual, predicted):
    c.execute("INSERT INTO predictions (actual_value, predicted_value) VALUES (?, ?)", (actual, predicted))
    conn.commit()
