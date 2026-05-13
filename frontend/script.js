const cards = document.querySelectorAll(".card");

cards.forEach(card => {

  card.addEventListener("mousemove", e => {

    const rect = card.getBoundingClientRect();

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const rotateY = (x - rect.width / 2) / 20;
    const rotateX = -(y - rect.height / 2) / 20;

    card.style.transform =
      `rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;

  });

  card.addEventListener("mouseleave", () => {

    if(card.classList.contains("card1")){
      card.style.transform = "rotate(-12deg)";
    }else{
      card.style.transform = "rotate(12deg)";
    }

  });

});