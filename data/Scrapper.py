import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By

#links = ['https://www.patterndesigns.com/en/gallery/2/Trends', 'https://www.patterndesigns.com/en/gallery/255/Timeless',
 #        'https://www.patterndesigns.com/en/gallery/350/Kids', 'https://www.patterndesigns.com/en/gallery/305/Floral',
 #        'https://www.patterndesigns.com/en/gallery/362/Animals', 'https://www.patterndesigns.com/en/gallery/299/Cultures',
 #        'https://www.patterndesigns.com/en/gallery/278/Seasonal', 'https://www.patterndesigns.com/en/gallery/313/Shapes',
 #        'https://www.patterndesigns.com/en/gallery/388/Tasty'
 #        ]

links = ['https://www.patterndesigns.com/en/gallery/388/Tasty'
         ]

# set up the webdriver
driver = webdriver.Chrome()
driver.maximize_window()

for link in links:
    driver.get(link)
    # wait for the page to load
    time.sleep(3)

    # scroll down until all the images are loaded
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(8)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    # find all images with the text "Mixed Designs"
    mixed_design_images = driver.find_elements(By.XPATH, '//*[@id="searchResults"]/div[2]/a/img')
    
    # download each image and save it with the title in the img tag
    for image in mixed_design_images:
        image_url = image.get_attribute('src')
        image_title = image.get_attribute('title')
        image_title = image_title.replace('/', '_')  # replace slashes in title with underscores
        
        with open(f'{image_title}.jpg', 'wb') as f:
            f.write(requests.get(image_url).content)

# close the webdriver
driver.quit()