from os import environ,system
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, request, session, redirect, url_for
from datetime import datetime, timedelta
import io
import base64
import bcrypt
import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from binance.client import Client

app = Flask(__name__)

environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def model_definition():
    model= Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(1,1)))
    model.add(LSTM(256))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

eth_model = model_definition()
eth_model.load_weights("models/eth_model.h5")
eth_model.make_predict_function()

btc_model = model_definition()
btc_model.load_weights("models/btc_model.h5")
btc_model.make_predict_function()

ltc_model = model_definition()
ltc_model.load_weights("models/ltc_model.h5")
ltc_model.make_predict_function()

xrp_model = model_definition()
xrp_model.load_weights("models/xrp_model.h5")
xrp_model.make_predict_function()

def data_extract(sym, start, end):
    client = Client('8ee716d569eaaac2dce8a4ddea9f9a3d0be43fede96c678b88e555ffbe80f930', 'db4d4ed5bb8da80dd90efa972cc279a79961f08e17921010508bcbd48be5eaf5')
    if end =="":
        cryptocurrency = client.get_historical_klines(symbol=sym, interval=Client.KLINE_INTERVAL_30MINUTE, start_str=start)
    else:
        cryptocurrency = client.get_historical_klines(symbol=sym, interval=Client.KLINE_INTERVAL_30MINUTE, start_str=start, end_str=end)

    cryptocurrency = pd.DataFrame(cryptocurrency, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    cryptocurrency['Open time'] = pd.to_datetime(cryptocurrency['Open time'], unit='ms')
    cryptocurrency.set_index('Open time', inplace=True)
    #print(CRYPTOCURRENCY.head())
    return cryptocurrency.iloc[:,3:4].astype(float).values

def plot_graph(crypto, predicted_price, real_price):
    img = io.BytesIO()
    plt.figure(figsize=(10,4))
    red_patch = mpatches.Patch(color='orange', label='Predicted Price of {}'.format(crypto))
    blue_patch = mpatches.Patch(color='purple', label='Real Price of {}'.format(crypto))
    plt.legend(handles=[blue_patch, red_patch])
    plt.plot(predicted_price, color='orange', label='Predicted Price of {}'.format(crypto))
    plt.plot(real_price, color='purple', label='Predicted Price of {}'.format(crypto))
    plt.title('Predicted vs. Real Price of {}'.format(crypto))
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    #plt.savefig('{}.png'.format(crypto))
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == "POST":
        crypto = request.form.get('crypto')
        start = request.form.get('start')
        end = request.form.get('end')
        investment = float(request.form.get('investment'))  # Get investment amount

        if crypto == "bitcoin":
            sym = "BTCUSDT"
            model = btc_model
        elif crypto == "ethereum":
            sym = "ETHUSDT"
            model = eth_model
        elif crypto == "ripple":
            sym = "XRPUSDT"
            model = xrp_model
        elif crypto == "litecoin":
            sym = "LTCUSDT"
            model = ltc_model
        else:
            print("Cryptocurrency not available")
            return render_template("index.html")

        data = data_extract(sym, start, end)
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        X_test = data_scaled[0:len(data_scaled) - 1]
        y_test = data_scaled[1:len(data_scaled)]
        X_test = np.reshape(X_test, (len(X_test), 1, X_test.shape[1]))

        predicted_prices_scaled = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predicted_prices_scaled)

        # Extract final predicted and real prices from arrays for today's prediction
        final_predicted_price = predicted_prices[-1][0]  # Today's prediction
        final_real_price = scaler.inverse_transform(y_test)[-1][0]  # Today's real price

        # Adjust predictions based on custom investment amount
        initial_price = scaler.inverse_transform([[data_scaled[-2][0]]])[0][0]
        num_coins = investment / initial_price  # Calculate number of coins bought

        # Calculate expected profit or loss based on the investment
        final_predicted_value = final_predicted_price * num_coins
        final_real_value = final_real_price * num_coins

        percentage_difference_today = ((final_real_value - final_predicted_value) / final_predicted_value) * 100

        if percentage_difference_today > 0:
            result_color = "green"
            recommendation = "You may consider selling."
        elif percentage_difference_today < 0:
            result_color = "red"
            recommendation = "It's advisable to hold."
        else:
            result_color = "black"
            recommendation = "Hold for now."

        result_message_today = f"Today's Prediction: ${final_predicted_value:.2f}, Real Value: ${final_real_value:.2f}, " \
                               f"<span style='color:{result_color};'>{('Profit' if percentage_difference_today > 0 else 'Loss')}: {abs(percentage_difference_today):.2f}%</span><br>" \
                               f"<span>Recommendation: {recommendation}</span><br><br>"

        # Prepare future predictions for the next 7 days
        future_steps = 7
        last_known_data = data_scaled[-1]
        last_known_date = datetime.strptime(end, "%Y-%m-%d")
        future_predictions = []

        current_input = np.reshape(last_known_data, (1, 1, last_known_data.shape[0]))
        for i in range(future_steps):
            next_day_prediction_scaled = model.predict(current_input)
            next_day_prediction = scaler.inverse_transform(next_day_prediction_scaled)
            next_day_date = last_known_date + timedelta(days=i + 1)
            predicted_price = "{:.2f}".format(next_day_prediction[0][0] * num_coins ) # Adjusted for investment
            future_predictions.append((next_day_date.strftime("%d-%m-%Y"), i + 1, predicted_price))
            current_input = np.reshape(next_day_prediction_scaled, (1, 1, 1))

        # Generate plot with custom investment for the entire period
        p_url = plot_graph(crypto, predicted_prices*num_coins, scaler.inverse_transform(y_test)*num_coins)

        return render_template("predict.html", result_message=result_message_today, future_predictions=future_predictions,plot_url='data:image/png;base64,{}'.format(p_url))

    return render_template("index.html")


# Index page that is rendered for every web call
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/crypto")
def crypto():
    if 'user' in session:
        user = session['user']
        return render_template("crypto.html",user=user)
    else:
        return redirect(url_for('login'))

app.secret_key = 'your_secret_key'

# Configure MySQL connection
db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='crypto_users'
)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get form data
        username = request.form['username']
        password = request.form['password']
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        name = request.form['name']
        gender = request.form['gender']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']

        try:
            # Create cursor
            cursor = db.cursor()

            # Check if user already exists
            cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", (username, email))
            existing_user = cursor.fetchone()

            if existing_user:
                error = "Username or email already exists. Please choose a different one."
                return render_template('register.html', error=error)

            # Insert new user with hashed password
            insert_query = "INSERT INTO users (username, password, name, gender, email, phone, address) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            cursor.execute(insert_query, (username, hashed_password, name, gender, email, phone, address))

            # Commit to DB
            db.commit()

            return redirect(url_for('login'))

        except mysql.connector.Error as err:
            error = f"An error occurred: {err}"
            return render_template('register.html', error=error)

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        # If user is already logged in, redirect to index or display a message
        message = "You are already logged in. No new logins available."
        return render_template('login.html', message=message)

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        try:
            # Create cursor
            cursor = db.cursor()

            # Execute query to fetch user from database
            query = "SELECT * FROM users WHERE username = %s"
            cursor.execute(query, (username,))
            user = cursor.fetchone()

            if user:
                # Verify password
                hashed_password = user[2]  # Assuming password hash is stored in the second column
                if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
                    # Store user information in session
                    session['user'] = user
                    return redirect(url_for('crypto'))
                else:
                    error = 'Invalid credentials. Please try again.'
                    return render_template('login.html', error=error)
            else:
                error = 'Invalid credentials. Please try again.'
                return render_template('login.html', error=error)

        except mysql.connector.Error as err:
            error = f"An error occurred: {err}"
            return render_template('login.html', error=error)

    return render_template('login.html')

@app.route('/reset', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        username = request.form['username']
        new_password = request.form['password']
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())

        try:
            # Create cursor
            cursor = db.cursor()

            # Check if the username exists in the database
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()

            if user:
                # Update user's password
                cursor.execute("UPDATE users SET password = %s WHERE username = %s", (hashed_password, username))
                db.commit()
                cursor.close()
                message = 'Password reset successful. You can now log in with your new password.'
                return render_template('reset.html', message=message)
            else:
                error = 'Username not found. Please enter a valid username.'
                return render_template('reset.html', error=error)

        except mysql.connector.Error as err:
            error = f"An error occurred: {err}"
            return render_template('reset.html', error=error)

    return render_template('reset.html')

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0') 