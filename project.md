## Evaluating the relationship between environmental variables and MPI using feature ranking and machine learning regression techniques

 We applied an random forest regression model with feature ranking to understand the relationship between the model diagnostics and MPI within a Titan-like GCM simulation. Results show that the effect of dissipative heating played a higher role in predicting MPI compared to the disequilibrium at the air-sea interface.
***

## 1. Introduction 

The presence, intensification, and decay of a tropical cyclone has always been associated as an Earth-like phenomena due to the warm and moist atmosphere that exists within the tropics. Defined as a warm core, cyclonic system, tropical cyclones are characterized by strong surface winds, spiral rainbands, and a clear, centralized eye. Outside of its appearance, the conditions needed for tropical cyclogenesis are shown in Table 1 [1].

The first three conditions provide the ingredients for deep convection, the latter affects the daily occurrence of tropical cyclones. Once conditions are met, the path of tropical cyclones are governed by the mean flow of the atmosphere, typically in a west-northwestward direction. Thus, tropical cyclones are highly present in each hemisphere during the summer and fall months.

Outside of Earth, Saturn's largest moon, Titan, has shown promise as the only "Earth-like" body within our solar system. The reasoning behind this claim comes from Titan's ability to mimic Earth's tropical environment due its slower rotation rate [4], resulting in the meridional expansion of the Hadley cell. However, it deviates from Earth in two ways: the main condensable and geography. Titan's main condensable, methane, is a highly volatile compound that experience high evaporation rates compared to water at the current average temperature of ~90 K [5], allowing for a equivalence of 5 meters of liquid methane to exist within the mid-to-upper troposphere. Additionally, Titan's geography is similar to that of a terraplanet, where dry sand dunes occupied the surface between latitudes $40^{\circ}$ N/S. But the polar regions (i.e., above $40{^\circ}$) contain secluded lakes and seas that are filled with liquid methane and ethane. In the case of tropical cyclones, the potential for deep convection is also a key ingredient for potential cyclogenesis. Observations from Earth-based telescopes and satellite imagery have shown that a range of violent and quiescent cloud systems are possible on Titan, especially at the poles during summer [4]. But, from a theoretical view, the potential for deep convection can be quantified via convective available potential energy (or CAPE), the buoyancy that a moist air parcel can experience between the level of free convection and the equilibrium level. At Titan's current surface temperature of ~95 K, a recent study has shown that convective activity is possible at 1500 J/kg [6]. Thus, regions at Titan's poles where significant CAPE is present could provide the necessary buoyancy needed to get deep convection, which would invoke strong inflow and vertical velocities, leading to potential tropical cyclogenesis. However, Titan's average cloud coverage is 0.5%, compared to 35-70% for terrestrial regimes [4]. So, it begs the question: **Why is Titan's poles not a playground for tropical cyclone formation**? 

One could use large-scale favorability metrics that account for the necessary conditions that a tropical cyclone needs in order to form. One such metric that has been used in previous studies is maximum potential intensity, or MPI [1,2,6]. It's main purpose is to establish an upper-bound measurement of the gradient winds that a tropical cyclone can achieve based on its surrounding environmental conditions. Looking at Equation (1), MPI is made up of three terms on the right hand side. The first term represents a ratio between how much enthalpy (i.e., latent and sensible heat) is exchanged at the air-sea interface and how much drag is caused by friction. The second term represent an efficiency term that similar to a Carnot cycle, but it accounts for how well a tropical cyclone can convert heat energy into work that is needed to power the gradient winds. The third and last term is the disequilibrium at the air-sea interface, represented by the difference of enthalpy at the sea-surface for a saturated air parcel ($k_s$) and enthalpy at the top of the boundary layer for a unsaturated air parcel ($k_a$). The last term is extremely important because it's a derived representation of the maximum amount of potential energy that a tropical cyclone can use to power the gradient winds. With the usage of this powerful quantity, we can gain foundational understanding of how favorable Titan's tropical environment is tropical cyclogenesis. However, it is rather unclear which of the terms is the dominant driver for MPI on Titan since it differs from Earth in terms of surface temperature. 

To address this issue head-on, we will use feature ranking by way of machine learning regression models. Feature ranking or importance is a technique by which a model ranks a set of features (i.e., temperature, specific humidity) from the training data that it thinks is the most important to determining a desired output (i.e., MPI). For the training data, we will use a series of GCM simulations of a "Titan-like" terraplanet to properly simulate Titan-like planetary and environmental characteristics in an idealized fashion. 
  

## 2. Data

To understand the relationship between MPI and its associated environmental variables on Titan, I will be looking at a total of 50 days during the southern hemisphere summer from the Titan Atmospheric Model (TAM), a three-dimensional GCM that has the ability to simulate Titan-like planetary and environmental characteristics, such as the solar constant, orbital period, planetary radius, a methane-rich atmosphere, and methane as the main condensable [3]. One of the model diagnostic, surface temperature, is plotted in Figure 1. The model is fitted with a full physics suite, including a quasi-equilibrium moist convection scheme, large-scale condensation scheme, non-grey radiation scheme, and a surface model with a bucket hydrology scheme [3]. The model resolution is 64x128, with 25 pressure levels in the vertical. For training and testing purposes, we have to preprocess the data because the model output is spatial and temporal in nature, with dimensions in latitude, longitude, and time, which can be seen in the data frame in Figure 3. 


Since most regression models are compatible with two-dimensional arrays, we must change the shape of the dimensions for each feature that is needed. To do so, we first need to reconfigure the data into Pandas dataframes, which can be shown in code below. Outside of MPI as a target variable, a total of ten features will be used to train the model.

```python
import pandas as pd
#Create a list of dataframes
main_data = []
for x in range(50):
    df = pd.DataFrame(data={'MPI': mpi_dfs[x], 'Latitude': lat_reshape, 'Longitude':lon_reshape,'air_sea_diseq': asd_dfs[x], 'diss_heat': dh_dfs[x],'SSH_sea_surface':ssh_dfs[x],'SH_boundary_layer':sh_dfs[x],'SST':sst_dfs[x],'sat_specific_enthalpy(sea_surface)':ks_dfs[x],'specific_enthalpy(boundary_layer)':ka_dfs[x],'bl_temp':t1_dfs[x]})
    main_data.append(df)

```

We create two new variables in a new `for` loop, in which, `X_data_days` represents all of the features, excluding MPI. `Y_data_days` represents our target variable. From there, we reshaped the arrays to combine time, latitude, and longitude to create a unique indicator for the dataset. The following code shows the procedure:

```python
X_data_days = []
Y_data_days = []
for x in range(50):
    X_data = main_data[x].drop(['MPI'],axis=1).values
    Y_data = main_data[x]['MPI'].values
    Y_data = Y_data.reshape(-1,1)
    X_data_days.append(X_data)
    Y_data_days.append(Y_data)

#Reshape the arrays
#Have to combine the time and sample lengths to adhere to the 2-D array requirement
X_data_days = X_data_days.reshape(-1,X_data_days.shape[-1])
Y_data_days = Y_data_days.reshape(-1,Y_data_days.shape[-1])
```

After each dataframe is created over the 50-day timespan, we checked for linear or no-linear correlations between MPI and the individual features using Pandas `corr ()` function.  Out of the ten features, the efficiency (`diss_heat`) and the air-sea disequilibrium term (`air_sea_diseq`)  show pretty high correlation coefficients at 0.99. However, outside of the MPI terms, variables such as specific humidity at the sea-surface (`SH_boundary_layer`) and enthalpy at the boundary layer (`specific_enthalpy(boundary_layer)`) also share strong correlations to MPI, pointing to a potential dependence on these two variables when training and testing begins. 

## Methodology

For the model choice, we will be using a regression model since the dataset contains a list of features, along with the desired output in MPI. To narrow down the type of regression model, we trained and tested a total of three models: a linear regression model with ridge regularization, a decision tree regression model, and a random forest regression model. For each model, the dataset was split into 80% for the training set and 20% for the test set. This is shown by the following code: 

```python
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_data_days, Y_data_days, test_size=0.2, shuffle=False)
```

After splitting the dataset, we scaled each feature and the target variable due to the different ranges of values. Afterwards, we fitted and evaluated the model using a performance measure called root-mean squared error, as shown by Equation (2), which calculates the average difference between the actual and predicted value by the model. The main goal behind RMSE is to get the difference between actual and predicted values close to zero as possible.  The RMSE scores for each model is the following:

```
RMSE for linear regression with ridge regularization: 0.106
RMSE for decision tree regression: 0.104
RMSE for random forest regression: 0.099
```

The random forest regression model appears to be the best model for this particular dataset. This makes sense considering that MPI has environmental variables that may have linear or non-linear relationships between them, thus making it difficult for linear models to handle. For the decision tree, it does a better job at predicting MPI, but its slightly better than the linear model. Thus, the random forest appears to the best fit. 

To find the best hyperparameters for the model, we used a function called `GridSearchCV`, which will go through numerous iterations of model configurations, based on the range of hyperparameters the user gives it. Here the following code as an example:

```python
from sklearn.model_selection import GridSearchCV
rfr_model = RandomForestRegressor()
param = [{'n_estimators':[1,10,30,50,100],'max_depth':[None,10,20,30],'min_samples_split':[2,5,10],'min_samples_leaf':[1,2,4]}]
grid_search = GridSearchCV(rfr_model,param,cv=5,scoring='neg_root_mean_squared_error')
```
## Results

Figure 2 shows gridded data of MPI for the first test case. MPI is measured in m/s.


Figure 3 shows gridded data for MPI for the first predicted case (random forest regression). MPI is measured in m/s.


Figure 4 shows the difference between the test and predicted cases in gridded format. 


Figure 5 shows feature ranking for the random forest regression model. The x-axis show feature importance. The y-axis show the individual features used in training and testing.

{}

## Discussion

For the random forest model, the difference between Figures 2 and 3 are fairly small, with slight deviations in weaker MPI being predicted by the model in the northern pole, as shown by Figure 4. For the feature ranking in Figure 5, we can see that there is a clear relationship between MPI and its individual terms, tropical cyclone efficiency and air-sea disequilibrium. However, there are two main discoveries that were noticeable. First, the efficiency term appears to have a higher feature rank than the air-sea disequilibrium term, despite smaller values (i.e., 0 to 1) . This dependence on the efficiency term points to MPI correlation to surface temperature, by which when it rises, it causes a increase in saturation vapor pressure, leading to a subsequent increase in humidity and thus surface enthalpy, a key ingredient in increasing the air-sea disequilibrium term. Second, the other environmental variables in the feature ranking have a zero feature importance in determining the model output. One possible explanation is that since the individual terms (i.e., efficiency, air-sea disequilibrium) that make up MPI are counted for in the feature ranking, it's possible the model used those variables to create the predicted output. Future work would need to be done without the individual MPI terms to verify this claim. 
## Conclusion

From this work, we can see that there is a clear relationship between MPI and its individual terms, such as tropical cyclone efficiency and the disequilibrium at the air-sea interface. In particular, the efficiency term appears to have a higher feature rank compared to the disequilibrium term, despite much lower values ranging from 0 to 1, compared to 0 to 10,000 for the disequilibrium term. This discovery point to the fact of how much surface ($T_s$) and tropopause temperatures ($T_o$) are correlated to surface enthalpy. As surface temperatures rise, this leads to increase in saturation vapor pressure, which increases humidity and therefore enthalpy. However, it is still surprising that the other environmental parameters are not found in the feature ranking. A potential future project will be redo the methodology without the main MPI terms to see if the model can still properly calculate MPI.

## References
[1] Emanuel, Kerry. "Tropical cyclones." *Annual review of earth and planetary sciences* 31.1 (2003): 75-104.

[2] Komacek, Thaddeus D., Daniel R. Chavas, and Dorian S. Abbot. "Hurricane genesis is favorable on terrestrial exoplanets orbiting late-type M dwarf stars." The Astrophysical Journal 898.2 (2020): 115.

[3] Lora, Juan M., Jonathan I. Lunine, and Joellen L. Russell. "GCM simulations of Titan’s middle and lower atmosphere and comparison to observations." _Icarus_ 250 (2015): 516-528.

[4] Mitchell, Jonathan L., and Juan M. Lora. "The climate of Titan." *Annual Review of Earth and Planetary Sciences* 44 (2016): 353-380.

[5] Roe, Henry G. "Titan's methane weather." *Annual Review of Earth and Planetary Sciences* 40 (2012): 355-382.[5] 

[6] Tokano, Tetsuya. "Are tropical cyclones possible over Titan’s polar seas?." *Icarus* 223.2 (2013): 766-774.

[7] Seeley, Jacob T., and Robin D. Wordsworth. "Moist convection is most vigorous at intermediate atmospheric humidity." arXiv preprint arXiv:2301.03669 (2023).

[back](./)