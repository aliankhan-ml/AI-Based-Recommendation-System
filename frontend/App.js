import React, { useState } from "react";
import axios from "axios";
import './App.css';

const App = () => {
  const [inputType, setInputType] = useState("");
  const [inputValue, setInputValue] = useState("");
  const [similarImages, setSimilarImages] = useState([]);

  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };

  const handleSelectChange = (e) => {
    setInputType(e.target.value);
  };

  function Base64ToImage({ base64 }) {
    return <img src={`data:image/png;base64, ${base64}`} alt="base64img" />;
  }

  const convertToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = (error) => reject(error);
      reader.readAsDataURL(file);
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      let inputData;
      if (inputType === "image") {
        const file = e.target.elements.inputFile.files[0];
        const base64Image = await convertToBase64(file);
        const imageData = base64Image.substring(base64Image.indexOf(',') + 1);
        inputData = { image: imageData };
      } else {
        inputData = { text: inputValue };
      }

      const response = await axios.post('/similar_images', inputData);
      setSimilarImages(response.data.similar_images);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <>
      
    <div className="container">
      <h1 className="heading">AI Based Recommendation System</h1>
      <form onSubmit={handleSubmit} >
        <div>
        <label>
          Input Type:
        </label>
          <select value={inputType} onChange={handleSelectChange}>
            <option value="">Select Type</option>
            <option value="image">Image</option>
            <option value="text">Text</option>
          </select>
        </div>
        <div>
        {inputType === "image" && (
          <>
          <label>Input Value:</label>
            <input type="file" name="inputFile" accept="image/*" />
          </>
        )}
        {inputType === "text" && (
          <>
            <label>Input Value:</label>
            <input
              type="text"
              value={inputValue}
              onChange={handleInputChange}
            />
          </>
        )}
        </div>
        <button type="submit">Submit</button>
      </form>

      {similarImages.length > 0 ? (
        <ul>
          {similarImages.map((image, index) => (
            <li key={index}>
              <Base64ToImage base64={image} />
            </li>
          ))}
        </ul>
      ) : (
        <p>No similar images found.</p>
      )}
    </div>
    </>
  );
};

export default App;
