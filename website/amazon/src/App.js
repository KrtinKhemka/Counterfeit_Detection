import "./App.css";
import Header from "./Header";
import Home from "./Home";
import Productlisting from "./Productlisting";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import React, { useState, useEffect } from "react";

function App() {
  const [data, setData] = useState("");

  useEffect(() => {
    fetch("http://127.0.0.1:8080/")
      .then((res) => res.text())
      .then((data) => {
        setData(data);
        console.log(data);
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
      });
  }, []);
  return (
    //BEM
    <Router>
      <div className="app">
        <Header />
        <Routes>
          <Route
            path="productlisting"
            element={<Productlisting Data={data} />}
          />
          <Route path="/" element={<Home />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
