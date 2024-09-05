var lingue = ee.FeatureCollection("users/cristianvergaraf/lingue"),
    region = 
    /* color: #d63000 */
    /* shown: false */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[-73.4125203613281, -39.16554642059811],
          [-73.4125203613281, -39.65150771482688],
          [-72.5885457519531, -39.65150771482688],
          [-72.5885457519531, -39.16554642059811]]], null, false);


function maskL8sr(image) {
  // Bit 0 - Fill
  // Bit 1 - Dilated Cloud
  // Bit 2 - Cirrus
  // Bit 3 - Cloud
  // Bit 4 - Cloud Shadow
  
  var qaMask = image.select('QA_PIXEL').bitwiseAnd(parseInt('11111', 2)).eq(0);
  var saturationMask = image.select('QA_RADSAT').eq(0);

  // Apply the scaling factors to the appropriate bands.
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);

  // Replace the original bands with the scaled ones and apply the masks.
  return image.addBands(opticalBands, null, true)
      .addBands(thermalBands, null, true)
      .updateMask(qaMask)
      .updateMask(saturationMask);
}

Map.setCenter(-73.2079, -39.4383, 10);  // 

/// Map the function over one year of data

var collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                     .filterDate('2020-01-01', '2022-11-17')
                     .filterBounds(lingue)
                     .map(maskL8sr);
                     
                     
                     
var collection_2021_summer = collection.filterDate('2020-12-21', '2021-03-21');
var collection_2021_fall = collection.filterDate('2021-03-22', '2021-06-21');
var collection_2021_winter = collection.filterDate('2021-06-21', '2021-09-21');
var collection_2021_spring = collection.filterDate('2021-09-21', '2021-12-21');                     

print("Imagenes 2021 summer", collection_2021_summer.size())
print("Imagenes 2021 fall", collection_2021_fall.size())
print("Imagenes 2021 winter", collection_2021_winter.size())
print("Imagenes 2021 spring", collection_2021_spring.size())


var composite_2021_summer = collection_2021_summer.median()
.clip(lingue);
var composite_2021_fall = collection_2021_fall.median()
.clip(lingue);
var composite_2021_winter = collection_2021_winter.median()
.clip(lingue);
var composite_2021_spring = collection_2021_spring.median()
.clip(lingue);     

Map.setCenter(-73.2079, -39.4383, 10);  // 

var bands_display = ['SR_B5', 'SR_B6', 'SR_B4']





Map.addLayer(composite_2021_summer, {bands: bands_display, min: 0, max: 0.4}, 'summer 2021');
Map.addLayer(composite_2021_fall, {bands: bands_display, min: 0, max: 0.4}, 'fall 2021');
Map.addLayer(composite_2021_winter, {bands: bands_display, min: 0, max: 0.4}, 'winter 2021');
Map.addLayer(composite_2021_spring, {bands: bands_display, min: 0, max: 0.4}, 'spring 2021');

// Export the image to Google Drive
//Export.image.toDrive({
//  image: composite_2021_summer,
 // description: 'composite_2021_summer',
//  scale: 30,  // Set the scale in meters
 // region: region // Set the region of interest
//});
