import React from "react";
import "./Product.css";

function Product() {
  return (
    <div className="product">
      <div className="product_info">
        <p>Product title</p>
        <p className="product_price">
          <small>$</small>
          <strong>129</strong>
        </p>
        <div className="product_rating">
          <p>⭐️</p>
        </div>
      </div>
      <img
        className="product_img"
        src="https://assets.bosecreative.com/transform/d3eff9c4-3559-4155-b5ac-acebb58c4456/QCH24_Black_001_RGB.png"
      />
      <button>Add to Basket</button>
    </div>
  );
}

export default Product;
