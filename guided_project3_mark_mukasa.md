# Guided Project: Exploring Ebay Car Sales Data

## Introduction
The dataset provided is of used cars from a classifieds section of the German Ebay website, the Kleinanzeigen. The dataset is 50,000 data points big.
The project is aimed at data cleaning and analysis the used car listings.

The data dictionary provided with data is as follows:

- dateCrawled - When this ad was first crawled. All field-values are taken from this date.
- name - Name of the car.
- seller - Whether the seller is private or a dealer.
- offerType - The type of listing
- price - The price on the ad to sell the car.
- abtest - Whether the listing is included in an A/B test.
- vehicleType - The vehicle Type.
- yearOfRegistration - The year in which the car was first registered.
- gearbox - The transmission type.
- powerPS - The power of the car in PS.
- model - The car model name.
- kilometer - How many kilometers the car has driven.
- monthOfRegistration - The month in which the car was first registered.
- fuelType - What type of fuel the car uses.
- brand - The brand of the car.
- notRepairedDamage - If the car has a damage which is not yet repaired.
- dateCreated - The date on which the eBay listing was created.
- nrOfPictures - The number of pictures in the ad.
- postalCode - The postal code for the location of the vehicle.
- lastSeenOnline - When the crawler saw this ad last online.





```python
import numpy as np
import pandas as pd
```

We'll read the dataset(csv file) into pandas. 


```python
autos = pd.read_csv('autos.csv', encoding = 'Latin-1')

```


```python
autos
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dateCrawled</th>
      <th>name</th>
      <th>seller</th>
      <th>offerType</th>
      <th>price</th>
      <th>abtest</th>
      <th>vehicleType</th>
      <th>yearOfRegistration</th>
      <th>gearbox</th>
      <th>powerPS</th>
      <th>model</th>
      <th>odometer</th>
      <th>monthOfRegistration</th>
      <th>fuelType</th>
      <th>brand</th>
      <th>notRepairedDamage</th>
      <th>dateCreated</th>
      <th>nrOfPictures</th>
      <th>postalCode</th>
      <th>lastSeen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2016-03-26 17:47:46</td>
      <td>Peugeot_807_160_NAVTECH_ON_BOARD</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$5,000</td>
      <td>control</td>
      <td>bus</td>
      <td>2004</td>
      <td>manuell</td>
      <td>158</td>
      <td>andere</td>
      <td>150,000km</td>
      <td>3</td>
      <td>lpg</td>
      <td>peugeot</td>
      <td>nein</td>
      <td>2016-03-26 00:00:00</td>
      <td>0</td>
      <td>79588</td>
      <td>2016-04-06 06:45:54</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2016-04-04 13:38:56</td>
      <td>BMW_740i_4_4_Liter_HAMANN_UMBAU_Mega_Optik</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$8,500</td>
      <td>control</td>
      <td>limousine</td>
      <td>1997</td>
      <td>automatik</td>
      <td>286</td>
      <td>7er</td>
      <td>150,000km</td>
      <td>6</td>
      <td>benzin</td>
      <td>bmw</td>
      <td>nein</td>
      <td>2016-04-04 00:00:00</td>
      <td>0</td>
      <td>71034</td>
      <td>2016-04-06 14:45:08</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2016-03-26 18:57:24</td>
      <td>Volkswagen_Golf_1.6_United</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$8,990</td>
      <td>test</td>
      <td>limousine</td>
      <td>2009</td>
      <td>manuell</td>
      <td>102</td>
      <td>golf</td>
      <td>70,000km</td>
      <td>7</td>
      <td>benzin</td>
      <td>volkswagen</td>
      <td>nein</td>
      <td>2016-03-26 00:00:00</td>
      <td>0</td>
      <td>35394</td>
      <td>2016-04-06 20:15:37</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2016-03-12 16:58:10</td>
      <td>Smart_smart_fortwo_coupe_softouch/F1/Klima/Pan...</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$4,350</td>
      <td>control</td>
      <td>kleinwagen</td>
      <td>2007</td>
      <td>automatik</td>
      <td>71</td>
      <td>fortwo</td>
      <td>70,000km</td>
      <td>6</td>
      <td>benzin</td>
      <td>smart</td>
      <td>nein</td>
      <td>2016-03-12 00:00:00</td>
      <td>0</td>
      <td>33729</td>
      <td>2016-03-15 03:16:28</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2016-04-01 14:38:50</td>
      <td>Ford_Focus_1_6_Benzin_TÜV_neu_ist_sehr_gepfleg...</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$1,350</td>
      <td>test</td>
      <td>kombi</td>
      <td>2003</td>
      <td>manuell</td>
      <td>0</td>
      <td>focus</td>
      <td>150,000km</td>
      <td>7</td>
      <td>benzin</td>
      <td>ford</td>
      <td>nein</td>
      <td>2016-04-01 00:00:00</td>
      <td>0</td>
      <td>39218</td>
      <td>2016-04-01 14:38:50</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>49995</td>
      <td>2016-03-27 14:38:19</td>
      <td>Audi_Q5_3.0_TDI_qu._S_tr.__Navi__Panorama__Xenon</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$24,900</td>
      <td>control</td>
      <td>limousine</td>
      <td>2011</td>
      <td>automatik</td>
      <td>239</td>
      <td>q5</td>
      <td>100,000km</td>
      <td>1</td>
      <td>diesel</td>
      <td>audi</td>
      <td>nein</td>
      <td>2016-03-27 00:00:00</td>
      <td>0</td>
      <td>82131</td>
      <td>2016-04-01 13:47:40</td>
    </tr>
    <tr>
      <td>49996</td>
      <td>2016-03-28 10:50:25</td>
      <td>Opel_Astra_F_Cabrio_Bertone_Edition___TÜV_neu+...</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$1,980</td>
      <td>control</td>
      <td>cabrio</td>
      <td>1996</td>
      <td>manuell</td>
      <td>75</td>
      <td>astra</td>
      <td>150,000km</td>
      <td>5</td>
      <td>benzin</td>
      <td>opel</td>
      <td>nein</td>
      <td>2016-03-28 00:00:00</td>
      <td>0</td>
      <td>44807</td>
      <td>2016-04-02 14:18:02</td>
    </tr>
    <tr>
      <td>49997</td>
      <td>2016-04-02 14:44:48</td>
      <td>Fiat_500_C_1.2_Dualogic_Lounge</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$13,200</td>
      <td>test</td>
      <td>cabrio</td>
      <td>2014</td>
      <td>automatik</td>
      <td>69</td>
      <td>500</td>
      <td>5,000km</td>
      <td>11</td>
      <td>benzin</td>
      <td>fiat</td>
      <td>nein</td>
      <td>2016-04-02 00:00:00</td>
      <td>0</td>
      <td>73430</td>
      <td>2016-04-04 11:47:27</td>
    </tr>
    <tr>
      <td>49998</td>
      <td>2016-03-08 19:25:42</td>
      <td>Audi_A3_2.0_TDI_Sportback_Ambition</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$22,900</td>
      <td>control</td>
      <td>kombi</td>
      <td>2013</td>
      <td>manuell</td>
      <td>150</td>
      <td>a3</td>
      <td>40,000km</td>
      <td>11</td>
      <td>diesel</td>
      <td>audi</td>
      <td>nein</td>
      <td>2016-03-08 00:00:00</td>
      <td>0</td>
      <td>35683</td>
      <td>2016-04-05 16:45:07</td>
    </tr>
    <tr>
      <td>49999</td>
      <td>2016-03-14 00:42:12</td>
      <td>Opel_Vectra_1.6_16V</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$1,250</td>
      <td>control</td>
      <td>limousine</td>
      <td>1996</td>
      <td>manuell</td>
      <td>101</td>
      <td>vectra</td>
      <td>150,000km</td>
      <td>1</td>
      <td>benzin</td>
      <td>opel</td>
      <td>nein</td>
      <td>2016-03-13 00:00:00</td>
      <td>0</td>
      <td>45897</td>
      <td>2016-04-06 21:18:48</td>
    </tr>
  </tbody>
</table>
<p>50000 rows × 20 columns</p>
</div>




```python
autos.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 50000 entries, 0 to 49999
    Data columns (total 20 columns):
    dateCrawled            50000 non-null object
    name                   50000 non-null object
    seller                 50000 non-null object
    offerType              50000 non-null object
    price                  50000 non-null object
    abtest                 50000 non-null object
    vehicleType            44905 non-null object
    yearOfRegistration     50000 non-null int64
    gearbox                47320 non-null object
    powerPS                50000 non-null int64
    model                  47242 non-null object
    odometer               50000 non-null object
    monthOfRegistration    50000 non-null int64
    fuelType               45518 non-null object
    brand                  50000 non-null object
    notRepairedDamage      40171 non-null object
    dateCreated            50000 non-null object
    nrOfPictures           50000 non-null int64
    postalCode             50000 non-null int64
    lastSeen               50000 non-null object
    dtypes: int64(5), object(15)
    memory usage: 7.6+ MB
    


```python
print(autos.head(2))
```

               dateCrawled                                        name  seller  \
    0  2016-03-26 17:47:46            Peugeot_807_160_NAVTECH_ON_BOARD  privat   
    1  2016-04-04 13:38:56  BMW_740i_4_4_Liter_HAMANN_UMBAU_Mega_Optik  privat   
    
      offerType   price   abtest vehicleType  yearOfRegistration    gearbox  \
    0   Angebot  $5,000  control         bus                2004    manuell   
    1   Angebot  $8,500  control   limousine                1997  automatik   
    
       powerPS   model   odometer  monthOfRegistration fuelType    brand  \
    0      158  andere  150,000km                    3      lpg  peugeot   
    1      286     7er  150,000km                    6   benzin      bmw   
    
      notRepairedDamage          dateCreated  nrOfPictures  postalCode  \
    0              nein  2016-03-26 00:00:00             0       79588   
    1              nein  2016-04-04 00:00:00             0       71034   
    
                  lastSeen  
    0  2016-04-06 06:45:54  
    1  2016-04-06 14:45:08  
    

## Observations
This dataset is composed of a dataframe with 20 columns and 50000 rows of data. 
Most of the columns donot have null values, save for the notRepairedDamage, fuelType, model, gearbox, and vehicleType columns. 
5 columns are of integer datatype, and 15 are of object datatypes.
The column names are of camelcase.


```python
autos.columns
```




    Index(['dateCrawled', 'name', 'seller', 'offerType', 'price', 'abtest',
           'vehicleType', 'yearOfRegistration', 'gearbox', 'powerPS', 'model',
           'odometer', 'monthOfRegistration', 'fuelType', 'brand',
           'notRepairedDamage', 'dateCreated', 'nrOfPictures', 'postalCode',
           'lastSeen'],
          dtype='object')



First, we shall change the column names from the camelcase to snakecase format. This is because the snakecase format is easier to work with and preferred in python. Other changes shall be made to columns to shortening purposes.


```python
edited_column_names = ['date_crawled', 'name', 'seller', 'offer_type', 'price', 'abtest',
       'vehicle_type', 'registration_year', 'gearbox', 'power_ps', 'model',
       'odometer', 'registration_month', 'fuel_type', 'brand',
       'unrepaired_damage', 'ad_created', 'nr_of_pictures', 'postal_code',
       'last_seen']
```


```python
autos.columns = edited_column_names
```


```python
autos
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date_crawled</th>
      <th>name</th>
      <th>seller</th>
      <th>offer_type</th>
      <th>price</th>
      <th>abtest</th>
      <th>vehicle_type</th>
      <th>registration_year</th>
      <th>gearbox</th>
      <th>power_ps</th>
      <th>model</th>
      <th>odometer</th>
      <th>registration_month</th>
      <th>fuel_type</th>
      <th>brand</th>
      <th>unrepaired_damage</th>
      <th>ad_created</th>
      <th>nr_of_pictures</th>
      <th>postal_code</th>
      <th>last_seen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2016-03-26 17:47:46</td>
      <td>Peugeot_807_160_NAVTECH_ON_BOARD</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$5,000</td>
      <td>control</td>
      <td>bus</td>
      <td>2004</td>
      <td>manuell</td>
      <td>158</td>
      <td>andere</td>
      <td>150,000km</td>
      <td>3</td>
      <td>lpg</td>
      <td>peugeot</td>
      <td>nein</td>
      <td>2016-03-26 00:00:00</td>
      <td>0</td>
      <td>79588</td>
      <td>2016-04-06 06:45:54</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2016-04-04 13:38:56</td>
      <td>BMW_740i_4_4_Liter_HAMANN_UMBAU_Mega_Optik</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$8,500</td>
      <td>control</td>
      <td>limousine</td>
      <td>1997</td>
      <td>automatik</td>
      <td>286</td>
      <td>7er</td>
      <td>150,000km</td>
      <td>6</td>
      <td>benzin</td>
      <td>bmw</td>
      <td>nein</td>
      <td>2016-04-04 00:00:00</td>
      <td>0</td>
      <td>71034</td>
      <td>2016-04-06 14:45:08</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2016-03-26 18:57:24</td>
      <td>Volkswagen_Golf_1.6_United</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$8,990</td>
      <td>test</td>
      <td>limousine</td>
      <td>2009</td>
      <td>manuell</td>
      <td>102</td>
      <td>golf</td>
      <td>70,000km</td>
      <td>7</td>
      <td>benzin</td>
      <td>volkswagen</td>
      <td>nein</td>
      <td>2016-03-26 00:00:00</td>
      <td>0</td>
      <td>35394</td>
      <td>2016-04-06 20:15:37</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2016-03-12 16:58:10</td>
      <td>Smart_smart_fortwo_coupe_softouch/F1/Klima/Pan...</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$4,350</td>
      <td>control</td>
      <td>kleinwagen</td>
      <td>2007</td>
      <td>automatik</td>
      <td>71</td>
      <td>fortwo</td>
      <td>70,000km</td>
      <td>6</td>
      <td>benzin</td>
      <td>smart</td>
      <td>nein</td>
      <td>2016-03-12 00:00:00</td>
      <td>0</td>
      <td>33729</td>
      <td>2016-03-15 03:16:28</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2016-04-01 14:38:50</td>
      <td>Ford_Focus_1_6_Benzin_TÜV_neu_ist_sehr_gepfleg...</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$1,350</td>
      <td>test</td>
      <td>kombi</td>
      <td>2003</td>
      <td>manuell</td>
      <td>0</td>
      <td>focus</td>
      <td>150,000km</td>
      <td>7</td>
      <td>benzin</td>
      <td>ford</td>
      <td>nein</td>
      <td>2016-04-01 00:00:00</td>
      <td>0</td>
      <td>39218</td>
      <td>2016-04-01 14:38:50</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>49995</td>
      <td>2016-03-27 14:38:19</td>
      <td>Audi_Q5_3.0_TDI_qu._S_tr.__Navi__Panorama__Xenon</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$24,900</td>
      <td>control</td>
      <td>limousine</td>
      <td>2011</td>
      <td>automatik</td>
      <td>239</td>
      <td>q5</td>
      <td>100,000km</td>
      <td>1</td>
      <td>diesel</td>
      <td>audi</td>
      <td>nein</td>
      <td>2016-03-27 00:00:00</td>
      <td>0</td>
      <td>82131</td>
      <td>2016-04-01 13:47:40</td>
    </tr>
    <tr>
      <td>49996</td>
      <td>2016-03-28 10:50:25</td>
      <td>Opel_Astra_F_Cabrio_Bertone_Edition___TÜV_neu+...</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$1,980</td>
      <td>control</td>
      <td>cabrio</td>
      <td>1996</td>
      <td>manuell</td>
      <td>75</td>
      <td>astra</td>
      <td>150,000km</td>
      <td>5</td>
      <td>benzin</td>
      <td>opel</td>
      <td>nein</td>
      <td>2016-03-28 00:00:00</td>
      <td>0</td>
      <td>44807</td>
      <td>2016-04-02 14:18:02</td>
    </tr>
    <tr>
      <td>49997</td>
      <td>2016-04-02 14:44:48</td>
      <td>Fiat_500_C_1.2_Dualogic_Lounge</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$13,200</td>
      <td>test</td>
      <td>cabrio</td>
      <td>2014</td>
      <td>automatik</td>
      <td>69</td>
      <td>500</td>
      <td>5,000km</td>
      <td>11</td>
      <td>benzin</td>
      <td>fiat</td>
      <td>nein</td>
      <td>2016-04-02 00:00:00</td>
      <td>0</td>
      <td>73430</td>
      <td>2016-04-04 11:47:27</td>
    </tr>
    <tr>
      <td>49998</td>
      <td>2016-03-08 19:25:42</td>
      <td>Audi_A3_2.0_TDI_Sportback_Ambition</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$22,900</td>
      <td>control</td>
      <td>kombi</td>
      <td>2013</td>
      <td>manuell</td>
      <td>150</td>
      <td>a3</td>
      <td>40,000km</td>
      <td>11</td>
      <td>diesel</td>
      <td>audi</td>
      <td>nein</td>
      <td>2016-03-08 00:00:00</td>
      <td>0</td>
      <td>35683</td>
      <td>2016-04-05 16:45:07</td>
    </tr>
    <tr>
      <td>49999</td>
      <td>2016-03-14 00:42:12</td>
      <td>Opel_Vectra_1.6_16V</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>$1,250</td>
      <td>control</td>
      <td>limousine</td>
      <td>1996</td>
      <td>manuell</td>
      <td>101</td>
      <td>vectra</td>
      <td>150,000km</td>
      <td>1</td>
      <td>benzin</td>
      <td>opel</td>
      <td>nein</td>
      <td>2016-03-13 00:00:00</td>
      <td>0</td>
      <td>45897</td>
      <td>2016-04-06 21:18:48</td>
    </tr>
  </tbody>
</table>
<p>50000 rows × 20 columns</p>
</div>




```python
autos_desc = autos.describe(include = 'all' )
print(autos_desc)
```

                   date_crawled         name  seller offer_type  price abtest  \
    count                 50000        50000   50000      50000  50000  50000   
    unique                48213        38754       2          2   2357      2   
    top     2016-03-12 16:06:22  Ford_Fiesta  privat    Angebot     $0   test   
    freq                      3           78   49999      49999   1421  25756   
    mean                    NaN          NaN     NaN        NaN    NaN    NaN   
    std                     NaN          NaN     NaN        NaN    NaN    NaN   
    min                     NaN          NaN     NaN        NaN    NaN    NaN   
    25%                     NaN          NaN     NaN        NaN    NaN    NaN   
    50%                     NaN          NaN     NaN        NaN    NaN    NaN   
    75%                     NaN          NaN     NaN        NaN    NaN    NaN   
    max                     NaN          NaN     NaN        NaN    NaN    NaN   
    
           vehicle_type  registration_year  gearbox      power_ps  model  \
    count         44905       50000.000000    47320  50000.000000  47242   
    unique            8                NaN        2           NaN    245   
    top       limousine                NaN  manuell           NaN   golf   
    freq          12859                NaN    36993           NaN   4024   
    mean            NaN        2005.073280      NaN    116.355920    NaN   
    std             NaN         105.712813      NaN    209.216627    NaN   
    min             NaN        1000.000000      NaN      0.000000    NaN   
    25%             NaN        1999.000000      NaN     70.000000    NaN   
    50%             NaN        2003.000000      NaN    105.000000    NaN   
    75%             NaN        2008.000000      NaN    150.000000    NaN   
    max             NaN        9999.000000      NaN  17700.000000    NaN   
    
             odometer  registration_month fuel_type       brand unrepaired_damage  \
    count       50000        50000.000000     45518       50000             40171   
    unique         13                 NaN         7          40                 2   
    top     150,000km                 NaN    benzin  volkswagen              nein   
    freq        32424                 NaN     30107       10687             35232   
    mean          NaN            5.723360       NaN         NaN               NaN   
    std           NaN            3.711984       NaN         NaN               NaN   
    min           NaN            0.000000       NaN         NaN               NaN   
    25%           NaN            3.000000       NaN         NaN               NaN   
    50%           NaN            6.000000       NaN         NaN               NaN   
    75%           NaN            9.000000       NaN         NaN               NaN   
    max           NaN           12.000000       NaN         NaN               NaN   
    
                     ad_created  nr_of_pictures   postal_code            last_seen  
    count                 50000         50000.0  50000.000000                50000  
    unique                   76             NaN           NaN                39481  
    top     2016-04-03 00:00:00             NaN           NaN  2016-04-07 06:17:27  
    freq                   1946             NaN           NaN                    8  
    mean                    NaN             0.0  50813.627300                  NaN  
    std                     NaN             0.0  25779.747957                  NaN  
    min                     NaN             0.0   1067.000000                  NaN  
    25%                     NaN             0.0  30451.000000                  NaN  
    50%                     NaN             0.0  49577.000000                  NaN  
    75%                     NaN             0.0  71540.000000                  NaN  
    max                     NaN             0.0  99998.000000                  NaN  
    

Examination of the dataset has shown that the seller, and offer_type columns have largely one value and therfore deserve to be dropped.
The price  and odometer columns have numeric data stored as text and hence need to be cleaned.

We'll now clean the data in the price and odometer columns, by removing any non-numeric characters and coverting the column to numeric type.


```python
autos['price'] = autos['price'].str.replace('$', '')
autos['price'] = autos['price'].str.replace(',', '')
autos['price'] = autos['price'].astype(int)

autos['odometer'] = autos['odometer'].str.replace('km', '')
autos['odometer'] = autos['odometer'].str.replace(',', '')
autos['odometer'] = autos['odometer'].astype(int)


```

We'll have to convert the odometer column name to odometer_km to maintain the meaning of the values in the column.


```python
autos.rename({'odometer': 'odometer_km'}, axis = 1, inplace = True)
```


```python
autos['odometer_km'].unique().shape
```




    (13,)




```python
autos['odometer_km'].describe()
```




    count     50000.000000
    mean     125732.700000
    std       40042.211706
    min        5000.000000
    25%      125000.000000
    50%      150000.000000
    75%      150000.000000
    max      150000.000000
    Name: odometer_km, dtype: float64




```python
autos['odometer_km'].value_counts()
```




    150000    32424
    125000     5170
    100000     2169
    90000      1757
    80000      1436
    70000      1230
    60000      1164
    50000      1027
    5000        967
    40000       819
    30000       789
    20000       784
    10000       264
    Name: odometer_km, dtype: int64




```python
autos['price'].unique().shape
```




    (2357,)




```python
autos['price'].describe()
```




    count    5.000000e+04
    mean     9.840044e+03
    std      4.811044e+05
    min      0.000000e+00
    25%      1.100000e+03
    50%      2.950000e+03
    75%      7.200000e+03
    max      1.000000e+08
    Name: price, dtype: float64




```python
autos['price'].value_counts().sort_index(ascending = True).head(15)
```




    0     1421
    1      156
    2        3
    3        1
    5        2
    8        1
    9        1
    10       7
    11       2
    12       3
    13       2
    14       1
    15       2
    17       3
    18       1
    Name: price, dtype: int64




```python
autos['price'].median()
```




    2950.0




```python
autos['price'].value_counts().sort_index(ascending = True).tail(15)
```




    265000      1
    295000      1
    299000      1
    345000      1
    350000      1
    999990      1
    999999      2
    1234566     1
    1300000     1
    3890000     1
    10000000    1
    11111111    2
    12345678    3
    27322222    1
    99999999    1
    Name: price, dtype: int64




```python
autos = autos[autos['price'].between(1, 350001)]
```


```python
autos.shape
```




    (48565, 20)



After inspection of the price column, I noticed there were some outliers required to be removed. Using the value_counts() method i observed the sudden jump from the 350000 price to the next value, 999999. I considered the values after the 350000 value to be outliers and removed rows that contained those values.

We are left with a dataframe with 48565 rows, indicating that we eliminated 1435 rows.


```python
autos
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date_crawled</th>
      <th>name</th>
      <th>seller</th>
      <th>offer_type</th>
      <th>price</th>
      <th>abtest</th>
      <th>vehicle_type</th>
      <th>registration_year</th>
      <th>gearbox</th>
      <th>power_ps</th>
      <th>model</th>
      <th>odometer_km</th>
      <th>registration_month</th>
      <th>fuel_type</th>
      <th>brand</th>
      <th>unrepaired_damage</th>
      <th>ad_created</th>
      <th>nr_of_pictures</th>
      <th>postal_code</th>
      <th>last_seen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2016-03-26 17:47:46</td>
      <td>Peugeot_807_160_NAVTECH_ON_BOARD</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>5000</td>
      <td>control</td>
      <td>bus</td>
      <td>2004</td>
      <td>manuell</td>
      <td>158</td>
      <td>andere</td>
      <td>150000</td>
      <td>3</td>
      <td>lpg</td>
      <td>peugeot</td>
      <td>nein</td>
      <td>2016-03-26 00:00:00</td>
      <td>0</td>
      <td>79588</td>
      <td>2016-04-06 06:45:54</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2016-04-04 13:38:56</td>
      <td>BMW_740i_4_4_Liter_HAMANN_UMBAU_Mega_Optik</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>8500</td>
      <td>control</td>
      <td>limousine</td>
      <td>1997</td>
      <td>automatik</td>
      <td>286</td>
      <td>7er</td>
      <td>150000</td>
      <td>6</td>
      <td>benzin</td>
      <td>bmw</td>
      <td>nein</td>
      <td>2016-04-04 00:00:00</td>
      <td>0</td>
      <td>71034</td>
      <td>2016-04-06 14:45:08</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2016-03-26 18:57:24</td>
      <td>Volkswagen_Golf_1.6_United</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>8990</td>
      <td>test</td>
      <td>limousine</td>
      <td>2009</td>
      <td>manuell</td>
      <td>102</td>
      <td>golf</td>
      <td>70000</td>
      <td>7</td>
      <td>benzin</td>
      <td>volkswagen</td>
      <td>nein</td>
      <td>2016-03-26 00:00:00</td>
      <td>0</td>
      <td>35394</td>
      <td>2016-04-06 20:15:37</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2016-03-12 16:58:10</td>
      <td>Smart_smart_fortwo_coupe_softouch/F1/Klima/Pan...</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>4350</td>
      <td>control</td>
      <td>kleinwagen</td>
      <td>2007</td>
      <td>automatik</td>
      <td>71</td>
      <td>fortwo</td>
      <td>70000</td>
      <td>6</td>
      <td>benzin</td>
      <td>smart</td>
      <td>nein</td>
      <td>2016-03-12 00:00:00</td>
      <td>0</td>
      <td>33729</td>
      <td>2016-03-15 03:16:28</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2016-04-01 14:38:50</td>
      <td>Ford_Focus_1_6_Benzin_TÜV_neu_ist_sehr_gepfleg...</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>1350</td>
      <td>test</td>
      <td>kombi</td>
      <td>2003</td>
      <td>manuell</td>
      <td>0</td>
      <td>focus</td>
      <td>150000</td>
      <td>7</td>
      <td>benzin</td>
      <td>ford</td>
      <td>nein</td>
      <td>2016-04-01 00:00:00</td>
      <td>0</td>
      <td>39218</td>
      <td>2016-04-01 14:38:50</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>49995</td>
      <td>2016-03-27 14:38:19</td>
      <td>Audi_Q5_3.0_TDI_qu._S_tr.__Navi__Panorama__Xenon</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>24900</td>
      <td>control</td>
      <td>limousine</td>
      <td>2011</td>
      <td>automatik</td>
      <td>239</td>
      <td>q5</td>
      <td>100000</td>
      <td>1</td>
      <td>diesel</td>
      <td>audi</td>
      <td>nein</td>
      <td>2016-03-27 00:00:00</td>
      <td>0</td>
      <td>82131</td>
      <td>2016-04-01 13:47:40</td>
    </tr>
    <tr>
      <td>49996</td>
      <td>2016-03-28 10:50:25</td>
      <td>Opel_Astra_F_Cabrio_Bertone_Edition___TÜV_neu+...</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>1980</td>
      <td>control</td>
      <td>cabrio</td>
      <td>1996</td>
      <td>manuell</td>
      <td>75</td>
      <td>astra</td>
      <td>150000</td>
      <td>5</td>
      <td>benzin</td>
      <td>opel</td>
      <td>nein</td>
      <td>2016-03-28 00:00:00</td>
      <td>0</td>
      <td>44807</td>
      <td>2016-04-02 14:18:02</td>
    </tr>
    <tr>
      <td>49997</td>
      <td>2016-04-02 14:44:48</td>
      <td>Fiat_500_C_1.2_Dualogic_Lounge</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>13200</td>
      <td>test</td>
      <td>cabrio</td>
      <td>2014</td>
      <td>automatik</td>
      <td>69</td>
      <td>500</td>
      <td>5000</td>
      <td>11</td>
      <td>benzin</td>
      <td>fiat</td>
      <td>nein</td>
      <td>2016-04-02 00:00:00</td>
      <td>0</td>
      <td>73430</td>
      <td>2016-04-04 11:47:27</td>
    </tr>
    <tr>
      <td>49998</td>
      <td>2016-03-08 19:25:42</td>
      <td>Audi_A3_2.0_TDI_Sportback_Ambition</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>22900</td>
      <td>control</td>
      <td>kombi</td>
      <td>2013</td>
      <td>manuell</td>
      <td>150</td>
      <td>a3</td>
      <td>40000</td>
      <td>11</td>
      <td>diesel</td>
      <td>audi</td>
      <td>nein</td>
      <td>2016-03-08 00:00:00</td>
      <td>0</td>
      <td>35683</td>
      <td>2016-04-05 16:45:07</td>
    </tr>
    <tr>
      <td>49999</td>
      <td>2016-03-14 00:42:12</td>
      <td>Opel_Vectra_1.6_16V</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>1250</td>
      <td>control</td>
      <td>limousine</td>
      <td>1996</td>
      <td>manuell</td>
      <td>101</td>
      <td>vectra</td>
      <td>150000</td>
      <td>1</td>
      <td>benzin</td>
      <td>opel</td>
      <td>nein</td>
      <td>2016-03-13 00:00:00</td>
      <td>0</td>
      <td>45897</td>
      <td>2016-04-06 21:18:48</td>
    </tr>
  </tbody>
</table>
<p>48565 rows × 20 columns</p>
</div>




```python
autos['date_crawled'].str[:10].value_counts(normalize = True, dropna = False).sort_index(ascending = False)
```




    2016-04-07    0.001400
    2016-04-06    0.003171
    2016-04-05    0.013096
    2016-04-04    0.036487
    2016-04-03    0.038608
    2016-04-02    0.035478
    2016-04-01    0.033687
    2016-03-31    0.031834
    2016-03-30    0.033687
    2016-03-29    0.034099
    2016-03-28    0.034860
    2016-03-27    0.031092
    2016-03-26    0.032204
    2016-03-25    0.031607
    2016-03-24    0.029342
    2016-03-23    0.032225
    2016-03-22    0.032987
    2016-03-21    0.037373
    2016-03-20    0.037887
    2016-03-19    0.034778
    2016-03-18    0.012911
    2016-03-17    0.031628
    2016-03-16    0.029610
    2016-03-15    0.034284
    2016-03-14    0.036549
    2016-03-13    0.015670
    2016-03-12    0.036920
    2016-03-11    0.032575
    2016-03-10    0.032184
    2016-03-09    0.033090
    2016-03-08    0.033296
    2016-03-07    0.036014
    2016-03-06    0.014043
    2016-03-05    0.025327
    Name: date_crawled, dtype: float64



From the exploration of the date_crawled column, the highest percentage of ads were crawled on 3rd April, and the least percentage crawled on 7th April. In addition, the percentage of adds crawled seems to lie within similar range for most of the days. 
The ads were crawled from March 5th through April 7th.


```python
autos['ad_created'].str[:10].value_counts(normalize = True, dropna = False).sort_index(ascending = False)
```




    2016-04-07    0.001256
    2016-04-06    0.003253
    2016-04-05    0.011819
    2016-04-04    0.036858
    2016-04-03    0.038855
                    ...   
    2015-12-05    0.000021
    2015-11-10    0.000021
    2015-09-09    0.000021
    2015-08-10    0.000021
    2015-06-11    0.000021
    Name: ad_created, Length: 76, dtype: float64



The ads were created from June 6th 2015 to April 7th 2016. From the exploration, the number of ads created steadily increased from June 6th 2015 and hit the maximum on 4th April 2016. 


```python
autos['last_seen'].str[:10].value_counts(normalize = True, dropna = False).sort_index(ascending = False)
```




    2016-04-07    0.131947
    2016-04-06    0.221806
    2016-04-05    0.124761
    2016-04-04    0.024483
    2016-04-03    0.025203
    2016-04-02    0.024915
    2016-04-01    0.022794
    2016-03-31    0.023783
    2016-03-30    0.024771
    2016-03-29    0.022341
    2016-03-28    0.020859
    2016-03-27    0.015649
    2016-03-26    0.016802
    2016-03-25    0.019211
    2016-03-24    0.019767
    2016-03-23    0.018532
    2016-03-22    0.021373
    2016-03-21    0.020632
    2016-03-20    0.020653
    2016-03-19    0.015834
    2016-03-18    0.007351
    2016-03-17    0.028086
    2016-03-16    0.016452
    2016-03-15    0.015876
    2016-03-14    0.012602
    2016-03-13    0.008895
    2016-03-12    0.023783
    2016-03-11    0.012375
    2016-03-10    0.010666
    2016-03-09    0.009595
    2016-03-08    0.007413
    2016-03-07    0.005395
    2016-03-06    0.004324
    2016-03-05    0.001071
    Name: last_seen, dtype: float64



From the exploration of the last_seen column, the crawler last saw the majority of the ads on April 6th.


```python
autos['registration_year'].describe()
```




    count    48565.000000
    mean      2004.755421
    std         88.643887
    min       1000.000000
    25%       1999.000000
    50%       2004.000000
    75%       2008.000000
    max       9999.000000
    Name: registration_year, dtype: float64



From exploration of the registration_year column, there are some observations of note;
- The earliest year is 1000
- The latest year is 9999
This is definitely out of order. We would have to eliminate rows where the registration_year is out of the 19th century.


```python
autos['registration_year'].value_counts(normalize = True, dropna = False).sort_index(ascending = False).head(15)
```




    9999    0.000062
    9000    0.000021
    8888    0.000021
    6200    0.000021
    5911    0.000021
    5000    0.000082
    4800    0.000021
    4500    0.000021
    4100    0.000021
    2800    0.000021
    2019    0.000041
    2018    0.009678
    2017    0.028663
    2016    0.025121
    2015    0.008072
    Name: registration_year, dtype: float64




```python
autos['registration_year'].describe()
```




    count    48565.000000
    mean      2004.755421
    std         88.643887
    min       1000.000000
    25%       1999.000000
    50%       2004.000000
    75%       2008.000000
    max       9999.000000
    Name: registration_year, dtype: float64




```python
autos['registration_year'][autos['registration_year'].between(1960,2016)].describe()
```




    count    46634.000000
    mean      2002.969400
    std          6.930433
    min       1960.000000
    25%       1999.000000
    50%       2003.000000
    75%       2008.000000
    max       2016.000000
    Name: registration_year, dtype: float64



From the exploration of the registration_year column, I decided to use the 1960-2016 as the acceptable values since the numbers jumped drastically from 1960 onwards and it isn't plausible that a car could be registered after the ad has been created hence the upper bound of 2016. Using the describe() method, it is clear that the median 2003 doesnt change from the original dataset(2004).
This selection changes eliminates 2022 entries.


```python
autos = autos[autos['registration_year'].between(1960, 2016)]
```


```python
autos['registration_year'].value_counts(normalize = True, dropna = False)
```




    2000    0.067676
    2005    0.062958
    1999    0.062122
    2004    0.057962
    2003    0.057876
    2006    0.057254
    2001    0.056525
    2002    0.053309
    1998    0.050671
    2007    0.048827
    2008    0.047498
    2009    0.044710
    1997    0.041836
    2011    0.034803
    2010    0.034074
    1996    0.029442
    2012    0.028091
    1995    0.026311
    2016    0.026161
    2013    0.017219
    2014    0.014217
    1994    0.013488
    1993    0.009114
    2015    0.008406
    1992    0.007934
    1990    0.007441
    1991    0.007269
    1989    0.003731
    1988    0.002895
    1985    0.002037
    1980    0.001823
    1986    0.001544
    1987    0.001544
    1984    0.001094
    1983    0.001094
    1978    0.000944
    1982    0.000879
    1970    0.000815
    1979    0.000729
    1972    0.000708
    1981    0.000600
    1968    0.000558
    1967    0.000558
    1971    0.000558
    1974    0.000515
    1960    0.000493
    1973    0.000493
    1966    0.000472
    1977    0.000472
    1976    0.000450
    1969    0.000407
    1975    0.000386
    1965    0.000365
    1964    0.000257
    1963    0.000172
    1961    0.000129
    1962    0.000086
    Name: registration_year, dtype: float64



Exploring the remaining values, the distribution is normal, 2000 being the year with bigger number of car registrations.

### Exploring Price by Brand
We would like to understand the average price per brand in the the dataset. 


```python
autos['brand'].describe()
```




    count          46634
    unique            40
    top       volkswagen
    freq            9860
    Name: brand, dtype: object




```python
autos['brand'].value_counts()
```




    volkswagen        9860
    bmw               5136
    opel              5018
    mercedes_benz     4495
    audi              4041
    ford              3257
    renault           2200
    peugeot           1393
    fiat              1197
    seat               852
    skoda              765
    nissan             713
    mazda              709
    smart              661
    citroen            653
    toyota             593
    hyundai            468
    sonstige_autos     441
    volvo              427
    mini               409
    mitsubishi         384
    honda              366
    kia                330
    alfa_romeo         309
    porsche            285
    suzuki             277
    chevrolet          265
    chrysler           164
    dacia              123
    daihatsu           117
    jeep               106
    subaru             100
    land_rover          97
    saab                77
    jaguar              73
    daewoo              70
    trabant             64
    rover               62
    lancia              50
    lada                27
    Name: brand, dtype: int64




```python
top_20_brands = autos['brand'].value_counts().index[:20]
```

The brand column contains 40 unique car brands, with the top brand being Volkswagen. 
I have chosen to aggregate the brand data basing on the top 20 car brands as seen using the value_counts() method.


```python
brand_by_price = {}

for b in top_20_brands:
    car_bool = autos['brand'] == b
    car_count = autos[car_bool]['price'].count()
    total_price = autos[car_bool]['price'].sum()
    mean_price = total_price / car_count
    brand_by_price[b] = mean_price
```


```python
brand_by_price
```




    {'volkswagen': 5398.871196754564,
     'bmw': 8332.203855140187,
     'opel': 2974.688122758071,
     'mercedes_benz': 8565.086095661847,
     'audi': 9336.687453600594,
     'ford': 3713.0242554498004,
     'renault': 2475.7172727272728,
     'peugeot': 3094.0172290021537,
     'fiat': 2813.748538011696,
     'seat': 4402.389671361502,
     'skoda': 6375.1477124183,
     'nissan': 4743.40252454418,
     'mazda': 4112.596614950635,
     'smart': 3580.2239031770046,
     'citroen': 3761.957120980092,
     'toyota': 5167.091062394604,
     'hyundai': 5365.254273504273,
     'sonstige_autos': 12363.965986394558,
     'volvo': 4946.501170960188,
     'mini': 10613.459657701711}




```python
sorted(brand_by_price, key=brand_by_price.get)
```




    ['renault',
     'fiat',
     'opel',
     'peugeot',
     'smart',
     'ford',
     'citroen',
     'mazda',
     'seat',
     'nissan',
     'volvo',
     'toyota',
     'hyundai',
     'volkswagen',
     'skoda',
     'bmw',
     'mercedes_benz',
     'audi',
     'mini',
     'sonstige_autos']



From the aggregation, it is observed that the sonstige_autos brand has the highest average price, followed by the mini brand, Audi and Mercedes Benz.
The cheapest brand is Renault, followed by Fiat, Opel and Peugeot.

To understand the average mileage for the top 20 brands, I shall carry out an aggregation similar to that done for the average price. Deductions on links between average mileage and average price for those brands can then be made.


```python
autos.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date_crawled</th>
      <th>name</th>
      <th>seller</th>
      <th>offer_type</th>
      <th>price</th>
      <th>abtest</th>
      <th>vehicle_type</th>
      <th>registration_year</th>
      <th>gearbox</th>
      <th>power_ps</th>
      <th>model</th>
      <th>odometer_km</th>
      <th>registration_month</th>
      <th>fuel_type</th>
      <th>brand</th>
      <th>unrepaired_damage</th>
      <th>ad_created</th>
      <th>nr_of_pictures</th>
      <th>postal_code</th>
      <th>last_seen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2016-03-26 17:47:46</td>
      <td>Peugeot_807_160_NAVTECH_ON_BOARD</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>5000</td>
      <td>control</td>
      <td>bus</td>
      <td>2004</td>
      <td>manuell</td>
      <td>158</td>
      <td>andere</td>
      <td>150000</td>
      <td>3</td>
      <td>lpg</td>
      <td>peugeot</td>
      <td>nein</td>
      <td>2016-03-26 00:00:00</td>
      <td>0</td>
      <td>79588</td>
      <td>2016-04-06 06:45:54</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2016-04-04 13:38:56</td>
      <td>BMW_740i_4_4_Liter_HAMANN_UMBAU_Mega_Optik</td>
      <td>privat</td>
      <td>Angebot</td>
      <td>8500</td>
      <td>control</td>
      <td>limousine</td>
      <td>1997</td>
      <td>automatik</td>
      <td>286</td>
      <td>7er</td>
      <td>150000</td>
      <td>6</td>
      <td>benzin</td>
      <td>bmw</td>
      <td>nein</td>
      <td>2016-04-04 00:00:00</td>
      <td>0</td>
      <td>71034</td>
      <td>2016-04-06 14:45:08</td>
    </tr>
  </tbody>
</table>
</div>




```python
brand_mean_mileage = {}

for b in top_20_brands:
    car_bool = autos['brand'] == b
    car_count = autos[car_bool]['odometer_km'].count()
    total_mileage = autos[car_bool]['odometer_km'].sum()
    mean_mileage = total_mileage / car_count
    brand_mean_mileage[b] = mean_mileage
```


```python
brand_mean_mileage
```




    {'volkswagen': 128707.9107505071,
     'bmw': 132597.35202492212,
     'opel': 129342.3674770825,
     'mercedes_benz': 130919.91101223581,
     'audi': 129157.38678544914,
     'ford': 124399.7543751919,
     'renault': 128127.27272727272,
     'peugeot': 127153.62526920316,
     'fiat': 117121.9715956558,
     'seat': 121267.60563380281,
     'skoda': 110875.81699346405,
     'nissan': 118330.99579242637,
     'mazda': 124464.03385049365,
     'smart': 99326.77760968229,
     'citroen': 119724.34915773354,
     'toyota': 115944.35075885328,
     'hyundai': 106442.30769230769,
     'sonstige_autos': 91360.54421768707,
     'volvo': 138067.9156908665,
     'mini': 88105.13447432763}




```python
bbp_series = pd.Series(brand_by_price)
bmm_series = pd.Series(brand_mean_mileage)
```


```python
bbp_df = pd.DataFrame(bbp_series, columns = ['mean_price'])
```


```python
bbp_df['mean_mileage'] = bmm_series
```


```python
bbp_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_price</th>
      <th>mean_mileage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>volkswagen</td>
      <td>5398.871197</td>
      <td>128707.910751</td>
    </tr>
    <tr>
      <td>bmw</td>
      <td>8332.203855</td>
      <td>132597.352025</td>
    </tr>
    <tr>
      <td>opel</td>
      <td>2974.688123</td>
      <td>129342.367477</td>
    </tr>
    <tr>
      <td>mercedes_benz</td>
      <td>8565.086096</td>
      <td>130919.911012</td>
    </tr>
    <tr>
      <td>audi</td>
      <td>9336.687454</td>
      <td>129157.386785</td>
    </tr>
    <tr>
      <td>ford</td>
      <td>3713.024255</td>
      <td>124399.754375</td>
    </tr>
    <tr>
      <td>renault</td>
      <td>2475.717273</td>
      <td>128127.272727</td>
    </tr>
    <tr>
      <td>peugeot</td>
      <td>3094.017229</td>
      <td>127153.625269</td>
    </tr>
    <tr>
      <td>fiat</td>
      <td>2813.748538</td>
      <td>117121.971596</td>
    </tr>
    <tr>
      <td>seat</td>
      <td>4402.389671</td>
      <td>121267.605634</td>
    </tr>
    <tr>
      <td>skoda</td>
      <td>6375.147712</td>
      <td>110875.816993</td>
    </tr>
    <tr>
      <td>nissan</td>
      <td>4743.402525</td>
      <td>118330.995792</td>
    </tr>
    <tr>
      <td>mazda</td>
      <td>4112.596615</td>
      <td>124464.033850</td>
    </tr>
    <tr>
      <td>smart</td>
      <td>3580.223903</td>
      <td>99326.777610</td>
    </tr>
    <tr>
      <td>citroen</td>
      <td>3761.957121</td>
      <td>119724.349158</td>
    </tr>
    <tr>
      <td>toyota</td>
      <td>5167.091062</td>
      <td>115944.350759</td>
    </tr>
    <tr>
      <td>hyundai</td>
      <td>5365.254274</td>
      <td>106442.307692</td>
    </tr>
    <tr>
      <td>sonstige_autos</td>
      <td>12363.965986</td>
      <td>91360.544218</td>
    </tr>
    <tr>
      <td>volvo</td>
      <td>4946.501171</td>
      <td>138067.915691</td>
    </tr>
    <tr>
      <td>mini</td>
      <td>10613.459658</td>
      <td>88105.134474</td>
    </tr>
  </tbody>
</table>
</div>



From the dataframe above, it can be deduced that car brands with less average mileage, end up being more expensive as compared to those with more mileage.
In conclusion, for this specific car listing, the the higher the avearage mileage, the less the price.


```python

```
