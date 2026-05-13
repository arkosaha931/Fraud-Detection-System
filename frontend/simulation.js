

let currentPaymentType = "Google Pay";

function setPaymentType(type) {

    currentPaymentType = type;

    document.getElementById("type").value = type;
}



function removeActive() {

    let buttons =
        document.querySelectorAll(".method-btn");

    buttons.forEach(btn => {
        btn.classList.remove("active");
    });
}


function showUPI(event) {

    document.getElementById("upi-form").style.display =
        "block";

    document.getElementById("netbanking-form").style.display =
        "none";

    document.getElementById("cash-form").style.display =
        "none";

    document.getElementById("card-form").style.display =
        "none";

    removeActive();

    event.currentTarget.classList.add("active");

    setPaymentType("Google Pay");
}



function showNetBanking(event) {

    document.getElementById("upi-form").style.display =
        "none";

    document.getElementById("netbanking-form").style.display =
        "block";

    document.getElementById("cash-form").style.display =
        "none";

    document.getElementById("card-form").style.display =
        "none";

    removeActive();

    event.currentTarget.classList.add("active");

    setPaymentType("Net Banking");
}



function showCash(event) {

    document.getElementById("upi-form").style.display =
        "none";

    document.getElementById("netbanking-form").style.display =
        "none";

    document.getElementById("cash-form").style.display =
        "block";

    document.getElementById("card-form").style.display =
        "none";

    removeActive();

    event.currentTarget.classList.add("active");

    setPaymentType("Cash Withdraw");
}



function showCard(event) {

    document.getElementById("upi-form").style.display =
        "none";

    document.getElementById("netbanking-form").style.display =
        "none";

    document.getElementById("cash-form").style.display =
        "none";

    document.getElementById("card-form").style.display =
        "block";

    removeActive();

    event.currentTarget.classList.add("active");

    setPaymentType("Card");
}



async function checkTransaction() {

    const amount =
        document.getElementById("amount").value;

    const type =
        document.getElementById("type").value;

    let resultBox =
        document.getElementById("result");

    

    if (!amount || amount <= 0) {

        resultBox.innerHTML =
            "❌ Please enter a valid amount";

        resultBox.style.background =
            "#dc2626";

        resultBox.style.color =
            "white";

        resultBox.style.padding =
            "18px";

        resultBox.style.borderRadius =
            "14px";

        resultBox.style.marginTop =
            "20px";

        return;
    }


    resultBox.innerHTML =
        "⏳ Analyzing transaction...";

    resultBox.style.background =
        "#2563eb";

    resultBox.style.color =
        "white";

    resultBox.style.padding =
        "18px";

    resultBox.style.borderRadius =
        "14px";

    resultBox.style.marginTop =
        "20px";

    try {


        const response = await fetch(
            "http://127.0.0.1:5000/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({

                    amount: amount,

                    type: type,

                    timestamp:
                        new Date().toISOString()

                })
            }
        );

       
        if (!response.ok) {

            throw new Error(
                "Backend server error"
            );
        }

        const data =
            await response.json();

        if (data.status === "safe") {

            resultBox.innerHTML = `

                ✅ Transaction Approved

                <br><br>

                Fraud Probability:
                ${data.fraud_probability}

            `;

            resultBox.style.background =
                "#16a34a";
        }


        else if (
            data.status === "suspicious"
        ) {

            resultBox.innerHTML = `

                ⚠️ Suspicious Transaction

                <br><br>

                OTP Verification Required

                <br><br>

                Fraud Probability:
                ${data.fraud_probability}

            `;

            resultBox.style.background =
                "#eab308";

            resultBox.style.color =
                "#111";
        }



        else {

            resultBox.innerHTML = `

                🚨 Fraudulent Transaction

                <br><br>

                Transaction Blocked

                <br><br>

                Fraud Probability:
                ${data.fraud_probability}

            `;

            resultBox.style.background =
                "#dc2626";
        }

    }


    catch (error) {

        console.error(error);

        resultBox.innerHTML = `

            ❌ Unable to connect
            to FraudLens AI Server

            <br><br>

            Please check Flask backend

        `;

        resultBox.style.background =
            "#991b1b";

        resultBox.style.color =
            "white";
    }
}