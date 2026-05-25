let currentPaymentType = "Google Pay";

function setPaymentType(type) {
    currentPaymentType = type;
    document.getElementById("type").value = type;
}

function removeActive() {
    document.querySelectorAll(".method-btn").forEach(btn => {
        btn.classList.remove("active");
    });
}

function showUPI(event) {
    document.getElementById("upi-form").style.display        = "block";
    document.getElementById("netbanking-form").style.display = "none";
    document.getElementById("cash-form").style.display       = "none";
    document.getElementById("card-form").style.display       = "none";
    removeActive();
    event.currentTarget.classList.add("active");
    setPaymentType("Google Pay");
}

function showNetBanking(event) {
    document.getElementById("upi-form").style.display        = "none";
    document.getElementById("netbanking-form").style.display = "block";
    document.getElementById("cash-form").style.display       = "none";
    document.getElementById("card-form").style.display       = "none";
    removeActive();
    event.currentTarget.classList.add("active");
    setPaymentType("Net Banking");
}

function showCash(event) {
    document.getElementById("upi-form").style.display        = "none";
    document.getElementById("netbanking-form").style.display = "none";
    document.getElementById("cash-form").style.display       = "block";
    document.getElementById("card-form").style.display       = "none";
    removeActive();
    event.currentTarget.classList.add("active");
    setPaymentType("Cash Withdraw");
}

function showCard(event) {
    document.getElementById("upi-form").style.display        = "none";
    document.getElementById("netbanking-form").style.display = "none";
    document.getElementById("cash-form").style.display       = "none";
    document.getElementById("card-form").style.display       = "block";
    removeActive();
    event.currentTarget.classList.add("active");
    setPaymentType("Card");
}

// ==========================================
// STYLE HELPER
// ==========================================

function styleResult(box, bg, color = "white") {
    box.style.background    = bg;
    box.style.color         = color;
    box.style.padding       = "18px";
    box.style.borderRadius  = "14px";
    box.style.marginTop     = "20px";
    box.style.lineHeight    = "1.7";
    box.style.fontSize      = "15px";
}

// ==========================================
// MAIN TRANSACTION HANDLER
// ==========================================

async function checkTransaction() {

    const senderAccount   = document.getElementById("sender_account").value.trim();
    const receiverAccount = document.getElementById("receiver_account").value.trim();
    const pin             = document.getElementById("pin").value.trim();
    const amount          = document.getElementById("amount").value.trim();
    const type            = document.getElementById("type").value;

    const resultBox = document.getElementById("result");

    // ---- VALIDATION ----

    if (!senderAccount || !receiverAccount || !pin || !amount) {
        styleResult(resultBox, "#dc2626");
        resultBox.innerHTML = "❌ Please fill all fields";
        return;
    }

    // ---- LOADING ----

    styleResult(resultBox, "#2563eb");
    resultBox.innerHTML = "⏳ Analyzing transaction with FraudLens AI...";

    try {

        const response = await fetch("http://127.0.0.1:5000/predict", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                sender_account:   senderAccount,
                receiver_account: receiverAccount,
                pin:              pin,
                amount:           amount,
                type:             type,
                location:         "Kolkata",
                device_info:      navigator.userAgent,
                timestamp:        new Date().toISOString()
            })
        });

        const data = await response.json();

        // ---- FAILED (validation errors) ----

        if (data.status === "failed") {
            styleResult(resultBox, "#dc2626");
            resultBox.innerHTML = `❌ ${data.message}`;
            return;
        }

        // ---- ERROR ----

        if (data.status === "error") {
            styleResult(resultBox, "#7f1d1d");
            resultBox.innerHTML = `⚠️ Server error: ${data.message}`;
            return;
        }

        // ---- SAFE ----

        if (data.status === "safe") {

            styleResult(resultBox, "#15803d");

            resultBox.innerHTML = `
                ✅ <strong>Transaction Approved</strong>
                <br><br>
                💸 Amount Transferred : <strong>₹${parseFloat(amount).toLocaleString("en-IN")}</strong>
                <br>
                🏦 Updated Balance : <strong>₹${parseFloat(data.balance).toLocaleString("en-IN")}</strong>
                <br><br>
                🤖 Fraud Probability : ${(data.fraud_probability * 100).toFixed(2)}%
                <br>
                🧠 NN Score : ${(data.nn_score * 100).toFixed(2)}%
                &nbsp;|&nbsp;
                🌲 ISO Score : ${(data.iso_score * 100).toFixed(2)}%
                &nbsp;|&nbsp;
                📊 Feature Score : ${(data.feature_score * 100).toFixed(2)}%
            `;

            return;
        }

        // ---- SUSPICIOUS ----

        if (data.status === "suspicious") {

            styleResult(resultBox, "#a16207", "#111");

            resultBox.innerHTML = `
                ⚠️ <strong>Suspicious Transaction Detected</strong>
                <br><br>
                OTP verification has been triggered.
                Transaction is held pending confirmation.
                <br><br>
                💸 Amount : <strong>₹${parseFloat(amount).toLocaleString("en-IN")}</strong>
                <br>
                🤖 Fraud Probability : ${(data.fraud_probability * 100).toFixed(2)}%
                <br>
                🧠 NN Score : ${(data.nn_score * 100).toFixed(2)}%
                &nbsp;|&nbsp;
                🌲 ISO Score : ${(data.iso_score * 100).toFixed(2)}%
                &nbsp;|&nbsp;
                📊 Feature Score : ${(data.feature_score * 100).toFixed(2)}%
            `;

            return;
        }

        // ---- FRAUD ----

        if (data.status === "fraud") {

            styleResult(resultBox, "#991b1b");

            resultBox.innerHTML = `
                🚨 <strong>Fraudulent Transaction Blocked</strong>
                <br><br>
                This transaction has been flagged as fraudulent
                and has been blocked. No money was transferred.
                <br><br>
                💸 Attempted Amount : <strong>₹${parseFloat(amount).toLocaleString("en-IN")}</strong>
                <br>
                🏦 Your Balance Unchanged : <strong>₹${parseFloat(data.balance).toLocaleString("en-IN")}</strong>
                <br><br>
                🤖 Fraud Probability : ${(data.fraud_probability * 100).toFixed(2)}%
                <br>
                🧠 NN Score : ${(data.nn_score * 100).toFixed(2)}%
                &nbsp;|&nbsp;
                🌲 ISO Score : ${(data.iso_score * 100).toFixed(2)}%
                &nbsp;|&nbsp;
                📊 Feature Score : ${(data.feature_score * 100).toFixed(2)}%
            `;

            return;
        }

    } catch (error) {

        console.error(error);

        styleResult(resultBox, "#7f1d1d");
        resultBox.innerHTML = `
            ❌ Unable to connect to FraudLens AI Server.
            <br>Make sure the Flask server is running on port 5000.
        `;
    }
}