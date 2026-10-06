from fastapi import FastAPI, HTTPException
from stock import StockDataAnalysis, MODEL_PATH, EVAL_PATH
from fastapi.responses import StreamingResponse
import yfinance as yf
import joblib, os

try:
    data = yf.download(["AAPL", "MSFT", "GOOGL", "TSLA"],
    start="2020-01-01",
    end="2025-01-01").reset_index()

except Exception as e:  # noqa: BLE001
    print(f'The {e} error occured!')

stock_data = StockDataAnalysis(data)
stock_data.set_date()

#Creating the FastAPI object for the app
app = FastAPI(title = 'STONKS')

@app.get('/')
def home():
    return{'message' : 'WELCOME TO STONKS!'}

@app.get('/stonks/observations/')
def observations_endpoint():

    try:
        figure = stock_data.observations()

    except Exception as e:
        
        raise HTTPException(status_code = 500, detail = f"Failed to generate plot: {e}")

    return StreamingResponse(figure, media_type = "image/png")

@app.get('/stonks/price_predictions/')
def predictions_endpoint():

    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code = 404,
            detail = "No saved predictions found. Run stock.py directly first to train and save the model."
        )
    try:
        predictions = joblib.load(MODEL_PATH)

    except Exception as e:
        raise HTTPException(status_code = 500, detaIl = f'ERROR OCCURED : {e}')
    
    return {'predictions' : predictions.tolist()}

@app.get('/stonks/rscore/')
def rscore_endpoint():

    if not os.path.exists(EVAL_PATH):
        raise HTTPException(status_code = 404, detail="No saved evaluation data found. Run stock.py directly first.")

    try:
        eval_data = joblib.load(EVAL_PATH)
        r_score = stock_data.r_score(eval_data['y_test'], eval_data['predictions'])

    except Exception as e:
        raise HTTPException(status_code = 500, detail = f'ERROR OCUURED: {e}')

    return {'R_Score' : r_score}

@app.get('/stonks/actualvspredicted')
def actualvspredicted_endpoint():

    if not os.path.exists(EVAL_PATH):
        raise HTTPException(status_code=404, detail = "No saved evaluation data found. Run stock.py directly first.")
    
    try:
        eval_data = joblib.load(EVAL_PATH)
        results = stock_data.actualvspredicted_results(eval_data['y_test'], eval_data['predictions'])
        results_json = results.to_dict(orient = 'records')

    except Exception as e:
        raise HTTPException(status_code = 500, detail = f'ERROR OCCURES: {e}')

    return {'ActualvsPredicted' : results_json}

@app.get('/stonks/moving_averages/')
def moving_averages_endpoint():

    try:
        figure = stock_data.moving_averages()

    except Exception as e:
        
        raise HTTPException(status_code = 500, detail = f"Failed to generate plot: {e}")

    return StreamingResponse(figure, media_type = "image/png")

