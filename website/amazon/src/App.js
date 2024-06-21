import "./App.css";
import Header from "./Header";
import Home from "./Home";
import Productlisting from "./Productlisting";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

function App() {
  return (
    //BEM
    <Router>
      <div className="app">
        <Header />,
        <Routes>
          <Route path="productlisting" element={<Productlisting />} />
          <Route path="/" element={<Home />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
