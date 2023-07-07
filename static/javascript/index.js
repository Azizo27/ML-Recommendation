function validateAge(input) {
    const age = parseInt(input.value);

    if (isNaN(age) || age < 1 || age > 130) {
        document.getElementById("ageError").textContent = "Enter a valid age";
        input.classList.add("invalid");
    } else {
        document.getElementById("ageError").textContent = "";
        input.classList.remove("invalid");
    }
}

function validateSeniority(input) {
    const seniority = parseInt(input.value);

    if (isNaN(seniority) || age < 0) {
        document.getElementById("seniorityError").textContent = "Enter valid seniority";
        input.classList.add("invalid");
    } else {
        document.getElementById("seniorityError").textContent = "";
        input.classList.remove("invalid");
    }
}


function validateGrossIncome(input) {
    const gross_income = parseFloat(input.value);

    if (isNaN(gross_income) || gross_income < 0) {
        document.getElementById("gross_incomeError").textContent = "Enter valid gross income";
        input.classList.add("invalid");
    } else {
        document.getElementById("gross_incomeError").textContent = "";
        input.classList.remove("invalid");
    }
}
