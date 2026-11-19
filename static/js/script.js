document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('car-form');
  if (!form) return;

  const year = document.getElementById('year');
  year.max = new Date().getFullYear();

  form.addEventListener('submit', (event) => {
    if (!form.checkValidity()) {
      event.preventDefault();
      event.stopPropagation();
    }
    form.classList.add('was-validated');
  });

  form.addEventListener('reset', () => {
    form.classList.remove('was-validated');
    window.setTimeout(() => {
      form.querySelectorAll('.is-invalid').forEach((field) => field.classList.remove('is-invalid'));
    }, 0);
  });
});
