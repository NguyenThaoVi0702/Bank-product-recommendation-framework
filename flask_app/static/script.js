// Utility function to format numbers with commas
function formatNumberWithCommas(number) {
    return number.toLocaleString("en-US");
}

// Fetch and display the customer list
document.addEventListener("DOMContentLoaded", () => {
    fetch("/get-customers")
        .then((response) => response.json())
        .then((customers) => {
            const tableBody = document.querySelector("#customer-table tbody");
            tableBody.innerHTML = ""; 
            customers.forEach((customer) => {
                const row = document.createElement("tr");
                row.innerHTML = `
                    <td>${customer.id}</td>
                `;
                row.addEventListener("click", () => displayCustomerDetails(customer.id, row));
                tableBody.appendChild(row);
            });
        });
});

// Fetch and display customer details
function displayCustomerDetails(customerId, row) {
    fetch(`/get-customer/${customerId}`)
        .then((response) => response.json())
        .then((customer) => {
            document.getElementById("customer-id").innerHTML = `<strong>Customer ID:</strong> ${customer.id}`;
            document.getElementById("customer-age").innerHTML = `<strong>Age:</strong> ${customer.age}`;
            document.getElementById("customer-income").innerHTML = `<strong>Income:</strong> ${formatNumberWithCommas(customer.income)} VND`;
            document.getElementById("customer-segment").innerHTML = `<strong>Segment:</strong> ${customer.segment}`;
            document.getElementById("customer-avatar").src = "/static/avatar.png"; 

            const purchaseHistoryBody = document.getElementById("purchase-history-body");
            purchaseHistoryBody.innerHTML = ""; 


            for (const [product, amount] of Object.entries(customer.purchase_history)) {
                const row = document.createElement("tr");
                row.innerHTML = `
                    <td>${product}</td>
                    <td>${formatNumberWithCommas(amount)} VND</td>
                `;
                purchaseHistoryBody.appendChild(row);
            }


            const recommendedProductEl = document.getElementById("recommended-product");
            recommendedProductEl.innerHTML = `Recommended Product: ${customer.recommended_product}`;
            recommendedProductEl.classList.add("highlight"); 

 
            const tableRows = document.querySelectorAll("#customer-table tbody tr");
            tableRows.forEach(tr => tr.classList.remove("selected")); 
            row.classList.add("selected"); 
        });
}
