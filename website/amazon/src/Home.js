import React from "react";
import "./Home.css";
import Product from "./Product";

function Home() {
  return (
    <div className="Home">
      <div className="home-container">
        <img
          className="home-image"
          src="https://images-eu.ssl-images-amazon.com/images/G/02/digital/video/merch2016/Hero/Covid19/Generic/GWBleedingHero_ENG_COVIDUPDATE__XSite_1500x600_PV_en-GB._CB428684220_.jpg"
        />
        <div className="home_row">
          <Product
            title="In Ear Black-Red Earphones"
            price={49.99}
            image="https://m.media-amazon.com/images/I/615SYkkPyDL._AC_SL1500_.jpg"
            rating={3}
          />
          <Product
            title="Amazon Fire-Tv Stick"
            price={100}
            image="https://m.media-amazon.com/images/I/7120GaDFhxL._AC_SL1000_.jpg"
            rating={5}
          />
        </div>
        <div className="home_row">
          <Product
            title="Portable Sub-Woofer"
            price={4.99}
            image="https://m.media-amazon.com/images/I/71FER1UJhcL._AC_SL1500_.jpg "
            rating={1}
          />
          <Product
            title="In Ear Earphones Black-Green"
            price={49.99}
            image="https://m.media-amazon.com/images/I/618zves-P8L._AC_SL1500_.jpg "
            rating={4}
          />
          <Product
            title="Xu Direct In Line Headphones"
            price={24.99}
            image="https://m.media-amazon.com/images/I/71dtAOC-bLL._AC_SL1500_.jpg "
            rating={4}
          />
        </div>
        <div className="home_row">
          <Product
            title="Amazon Basics E-300 Headphones"
            price={70}
            image="https://m.media-amazon.com/images/I/71VHRNgvpqL._AC_SL1500_.jpg "
            rating={5}
          />
        </div>
      </div>
    </div>
  );
}

export default Home;
