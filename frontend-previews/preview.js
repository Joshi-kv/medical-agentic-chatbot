document.querySelectorAll(".composer").forEach((form) => {
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const input = form.querySelector("input");
    const toast = document.querySelector(".demo-toast");
    if (!input.value.trim()) return;
    toast.textContent = `Preview message: ${input.value.trim()}`;
    toast.classList.add("visible");
    input.value = "";
    window.setTimeout(() => toast.classList.remove("visible"), 2400);
  });
});
