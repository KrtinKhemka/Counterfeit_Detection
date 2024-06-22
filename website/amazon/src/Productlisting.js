import React, { useState, useEffect } from 'react';
import './Productlisting.css';
import { useParams } from 'react-router-dom';

function Productlisting() {
    const { Unique_product_id } = useParams();
    const [products, setProducts] = useState(null);

    useEffect(() => {
        fetch(`http://localhost:8000/products/${Unique_product_id}`)
            .then((response) => response.json())
            .then((data) => setProducts(data))
            .catch((error) => console.error("Error fetching data:", error));
    }, [Unique_product_id]);

    if (!products) {
        return <div>Loading...</div>;
    }
    const isFake = products[0].Product_score > 70;
    return (
        <div className="product-listing">
            
            <div className='product-detail'><h1>Product ID: {Unique_product_id}</h1>
            <img src={products[0].Photo_url} alt="Product" className="product-image" />
             <a href={products[0].Product_link} target="_blank" rel="noopener noreferrer" className="product-link">View Product</a>
            <p className='desc'><strong>Description:</strong> {products[0].Description}</p>
            <p><strong>Price:</strong> ${products[0].Price}</p>
               <p className={`prod-score ${isFake ? 'fake' : ''}`}>
              <strong>{isFake ? 'This product seems to be Fake!' : ''}</strong>
              <p>{`Product Score: ${products[0].Product_score}`}</p>
            </p>
            </div>
           
            {products.map((product) => (
                <div key={product._id} className="product-card">
                    
                    <div className="product-info">
                       
                        
                        
                        <p><strong>Review:</strong> {product.review_bold}</p>
                        <p>{product.review}</p>
                        <p><strong>Rating:</strong> {product.ratings} stars</p>
                        <p><strong>Review by:</strong> {product.by} on {product.date}</p>
                        <p><strong>Helpful Votes:</strong> {product.helpful}</p>
                        <p className='rev-score'><strong>Review Score:</strong> {product.FINAL_SCORE}</p>
                    </div>
                </div>
            ))}
        </div>
    );
}

export default Productlisting;
