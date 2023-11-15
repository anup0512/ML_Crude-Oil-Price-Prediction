 
import pickle
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly

# loading the trained model
pickle_in = open('prophet.pkl', 'rb') 
model = pickle.load(pickle_in)
 
@st.cache_data() #Decorator fn
  
# defining the function which will make the forecast:


def prediction(n_years):   
      # Making predictions 
    future = model.make_future_dataframe(periods = n_years*365)
    forecast = model.predict(future)
    data = forecast.iloc[n_years*(-365):]
    return data
      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:white;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Crude Oil Price Forecast</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True)
    
    #Input years:
    n_years = st.slider("Select the number of years of forecast: ",1,4)
      
    # When 'Forecast' is clicked, make the forecast:
    if st.button("Forecast"): 
        result = prediction(n_years)
        st.subheader('Forecast Data:')
        st.write(result[['ds','yhat']])
        st.subheader('Forecast Plot:')
        fig = plot_plotly(model,result)
        st.plotly_chart(fig)
        
if __name__=='__main__': 
    main()
