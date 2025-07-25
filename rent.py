import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pickle
import json

df2= pd.read_csv('Metro_House_Rent.csv')


# Filter the data to include only rows where the city is 'Bangalore'
bangalore_data = df2[df2['city'] == 'Bangalore']

# Display the resulting table
bangalore_data.reset_index(drop=True, inplace=True)
bangalore_data.head()

bangalore_data.shape

columns_to_drop = ['city', 'floor', 'animal_allowance', 'furniture', 'parking_spaces', 
                   'association_tax', 'rent_amount', 'property_tax', 'fire_insurance']
data = bangalore_data.drop(columns=columns_to_drop, errors='ignore')

data.shape

data.head()

areas_to_keep = ['1st Block Jayanagar', '1st Phase JP Nagar',
       '2nd Phase Judicial Layout', '2nd Stage Nagarbhavi',
       '5th Block Hbr Layout', '5th Phase JP Nagar', '6th Phase JP Nagar',
       '7th Phase JP Nagar', '8th Phase JP Nagar', '9th Phase JP Nagar',
       'AECS Layout', 'Abbigere', 'Akshaya Nagar', 'Ambalipura',
       'Ambedkar Nagar', 'Amruthahalli', 'Anandapura', 'Ananth Nagar',
       'Anekal', 'Anjanapura', 'Ardendale', 'Arekere', 'Attibele',
       'BEML Layout', 'BTM 2nd Stage', 'BTM Layout', 'Babusapalaya',
       'Badavala Nagar', 'Balagere', 'Banashankari',
       'Banashankari Stage II', 'Banashankari Stage III',
       'Banashankari Stage V', 'Banashankari Stage VI', 'Banaswadi',
       'Banjara Layout', 'Bannerghatta', 'Bannerghatta Road',
       'Basavangudi', 'Basaveshwara Nagar', 'Battarahalli', 'Begur',
       'Begur Road', 'Bellandur', 'Benson Town', 'Bharathi Nagar',
       'Bhoganhalli', 'Billekahalli', 'Binny Pete', 'Bisuvanahalli',
       'Bommanahalli', 'Bommasandra', 'Bommasandra Industrial Area',
       'Bommenahalli', 'Brookefield', 'Budigere', 'CV Raman Nagar',
       'Chamrajpet', 'Chandapura', 'Channasandra', 'Chikka Tirupathi',
       'Chikkabanavar', 'Chikkalasandra', 'Choodasandra', 'Cooke Town',
       'Cox Town', 'Cunningham Road', 'Dasanapura', 'Dasarahalli',
       'Devanahalli', 'Devarachikkanahalli', 'Dodda Nekkundi',
       'Doddaballapur', 'Doddakallasandra', 'Doddathoguru', 'Domlur',
       'Dommasandra', 'EPIP Zone', 'Electronic City',
       'Electronic City Phase II', 'Electronics City Phase 1',
       'Frazer Town', 'GM Palaya', 'Garudachar Palya', 'Giri Nagar',
       'Gollarapalya Hosahalli', 'Gottigere', 'Green Glen Layout',
       'Gubbalala', 'Gunjur', 'HAL 2nd Stage', 'HBR Layout',
       'HRBR Layout', 'HSR Layout', 'Haralur Road', 'Harlur', 'Hebbal',
       'Hebbal Kempapura', 'Hegde Nagar', 'Hennur', 'Hennur Road',
       'Hoodi', 'Horamavu Agara', 'Horamavu Banaswadi', 'Hormavu',
       'Hosa Road', 'Hosakerehalli', 'Hoskote', 'Hosur Road', 'Hulimavu',
       'ISRO Layout', 'ITPL', 'Iblur Village', 'Indira Nagar', 'JP Nagar',
       'Jakkur', 'Jalahalli', 'Jalahalli East', 'Jigani',
       'Judicial Layout', 'KR Puram', 'Kadubeesanahalli', 'Kadugodi',
       'Kaggadasapura', 'Kaggalipura', 'Kaikondrahalli',
       'Kalena Agrahara', 'Kalyan nagar', 'Kambipura', 'Kammanahalli',
       'Kammasandra', 'Kanakapura', 'Kanakpura Road', 'Kannamangala',
       'Karuna Nagar', 'Kasavanhalli', 'Kasturi Nagar', 'Kathriguppe',
       'Kaval Byrasandra', 'Kenchenahalli', 'Kengeri',
       'Kengeri Satellite Town', 'Kereguddadahalli', 'Kodichikkanahalli',
       'Kodigehaali', 'Kodigehalli', 'Kodihalli', 'Kogilu', 'Konanakunte',
       'Koramangala', 'Kothannur', 'Kothanur', 'Kudlu', 'Kudlu Gate',
       'Kumaraswami Layout', 'Kundalahalli', 'LB Shastri Nagar',
       'Laggere', 'Lakshminarayana Pura', 'Lingadheeranahalli',
       'Magadi Road', 'Mahadevpura', 'Mahalakshmi Layout', 'Mallasandra',
       'Malleshpalya', 'Malleshwaram', 'Marathahalli', 'Margondanahalli',
       'Marsur', 'Mico Layout', 'Munnekollal', 'Murugeshpalya',
       'Mysore Road', 'NGR Layout', 'NRI Layout', 'Nagarbhavi',
       'Nagasandra', 'Nagavara', 'Nagavarapalya', 'Narayanapura',
       'Neeladri Nagar', 'Nehru Nagar', 'OMBR Layout', 'Old Airport Road',
       'Old Madras Road', 'Padmanabhanagar', 'Pai Layout', 'Panathur',
       'Parappana Agrahara', 'Pattandur Agrahara', 'Poorna Pragna Layout',
       'Prithvi Layout', 'R.T. Nagar', 'Rachenahalli',
       'Raja Rajeshwari Nagar', 'Rajaji Nagar', 'Rajiv Nagar',
       'Ramagondanahalli', 'Ramamurthy Nagar', 'Rayasandra',
       'Sahakara Nagar', 'Sanjay nagar', 'Sarakki Nagar', 'Sarjapur',
       'Sarjapur  Road', 'Sarjapura - Attibele Road',
       'Sector 2 HSR Layout', 'Sector 7 HSR Layout', 'Seegehalli',
       'Shampura', 'Shivaji Nagar', 'Singasandra', 'Somasundara Palya',
       'Sompura', 'Sonnenahalli', 'Subramanyapura', 'Sultan Palaya',
       'TC Palaya', 'Talaghattapura', 'Thanisandra', 'Thigalarapalya',
       'Thubarahalli', 'Thyagaraja Nagar', 'Tindlu', 'Tumkur Road',
       'Ulsoor', 'Uttarahalli', 'Varthur', 'Varthur Road', 'Vasanthapura',
       'Vidyaranyapura', 'Vijayanagar', 'Vishveshwarya Layout',
       'Vishwapriya Layout', 'Vittasandra', 'Whitefield',
       'Yelachenahalli', 'Yelahanka', 'Yelahanka New Town', 'Yelenahalli',
       'Yeshwanthpur']


# Filter the DataFrame to keep only the rows where the 'Area' column matches the areas you want
data_filtered = data[data['area'].isin(areas_to_keep)]
# Save the filtered data to a new CSV file
data_filtered.to_csv("filtered_areas_dataset.csv", index=False)

#data['area'].unique()
len(data_filtered.area.unique())

data_filtered.head()
data_filtered.isnull().sum()
data_filtered = data_filtered.dropna()
df1 = pd.get_dummies(data_filtered, columns=['area'], drop_first=True)

X = df1.drop(columns=['total_rent'], errors='ignore')  # Features
y = df1['total_rent']

df1.head()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Test RMSE: {rmse}")

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
param_grid = {'fit_intercept': [True, False]}  # Example parameter grid
grid_search = GridSearchCV(LinearRegression(), param_grid, cv=cv, scoring='neg_mean_squared_error')
grid_search.fit(X_scaled, y)

print("Best Cross-Validation Score:", grid_search.best_score_)
print("Best Parameters:", grid_search.best_params_)

def predict_rent(area, rooms, bathrooms):
    # Create an array of zeros with the same number of features as X_train
    x = np.zeros(X.shape[1])
    
    # Set the values for the rooms and bathrooms
    x[0] = rooms  # Assuming 'rooms' is the first column
    x[1] = bathrooms  # Assuming 'bathrooms' is the second column

    # Handle one-hot encoding for 'area'
    area_column = f"area_{area}"
    if area_column in X.columns:
        loc_index = np.where(X.columns == area_column)[0][0]
        x[loc_index] = 1  # Set the specific area column to 1

    # Scale the input using the fitted scaler
    x_scaled = scaler.transform([x])

    # Predict the rent using the trained model
    predicted_rent = lr_model.predict(x_scaled)[0]
    return predicted_rent

# Example usage of prediction function
predicted_rent = predict_rent(area="Whitefield", rooms=3, bathrooms=2)
print(f"Predicted Rent for 3 BHK, 2 Bathrooms in Whitefield: {predicted_rent}")

predict_rent('1st Phase JP Nagar',2,2)

predict_rent('Yelenahalli',2,2)

with open('bangalore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)

columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))