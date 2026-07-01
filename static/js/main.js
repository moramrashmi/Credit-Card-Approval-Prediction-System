/**
 * main.js — Credit Card Approval System
 * =======================================
 * Handles:
 *   - Form validation with real-time feedback
 *   - Loading overlay during form submission
 *   - Scroll-triggered fade-up animations
 *   - Probability gauge animation on result page
 *   - Navbar scroll effect
 *   - Smooth scroll for anchor links
 */

"use strict";

/* ─── Utilities ──────────────────────────────────────────────── */

/**
 * Add "visible" class to elements with .fade-up when they enter the viewport.
 * Uses IntersectionObserver for performance — no scroll event listeners.
 */
function initScrollAnimations() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target); // animate once
        }
      });
    },
    { threshold: 0.12 }
  );

  document.querySelectorAll(".fade-up").forEach((el) => observer.observe(el));
}

/**
 * Shrink/highlight navbar on scroll.
 */
function initNavbarScroll() {
  const navbar = document.querySelector(".navbar");
  if (!navbar) return;

  window.addEventListener(
    "scroll",
    () => {
      navbar.classList.toggle("scrolled", window.scrollY > 40);
    },
    { passive: true }
  );
}

/* ─── Form Validation ─────────────────────────────────────────── */

/**
 * Client-side validation rules for the prediction form.
 * Server-side validation in app.py is the authoritative check;
 * this provides instant UX feedback without a round-trip.
 */
const FIELD_RULES = {
  Age: {
    label: "Age",
    validate: (v) => {
      const n = parseFloat(v);
      return !isNaN(n) && n >= 18 && n <= 100
        ? null
        : "Age must be between 18 and 100.";
    },
  },
  Debt: {
    label: "Debt",
    validate: (v) => {
      const n = parseFloat(v);
      return !isNaN(n) && n >= 0 ? null : "Debt must be a non-negative number.";
    },
  },
  YearsEmployed: {
    label: "Years Employed",
    validate: (v) => {
      const n = parseFloat(v);
      return !isNaN(n) && n >= 0 && n <= 60
        ? null
        : "Years Employed must be between 0 and 60.";
    },
  },
  CreditScore: {
    label: "Credit Score",
    validate: (v) => {
      const n = parseFloat(v);
      return !isNaN(n) && n >= 0 ? null : "Credit Score must be a non-negative number.";
    },
  },
  ZipCode: {
    label: "Zip Code",
    validate: (v) => {
      const n = parseFloat(v);
      return !isNaN(n) && n >= 0 ? null : "Zip Code must be a valid number.";
    },
  },
  Income: {
    label: "Income",
    validate: (v) => {
      const n = parseFloat(v);
      return !isNaN(n) && n >= 0 ? null : "Income must be a non-negative number.";
    },
  },
};

/**
 * Show or clear validation feedback for a single field.
 */
function setFieldValidity(field, errorMsg) {
  const existing = field.parentElement.querySelector(".field-error");
  if (existing) existing.remove();
  field.classList.remove("is-invalid", "is-valid");

  if (errorMsg) {
    field.classList.add("is-invalid");
    const div = document.createElement("div");
    div.className = "invalid-feedback field-error";
    div.textContent = errorMsg;
    field.after(div);
  } else {
    field.classList.add("is-valid");
  }
}

/**
 * Validate a single field on blur.
 */
function validateField(field) {
  const rule = FIELD_RULES[field.name];
  if (!rule) {
    // For select elements: just check non-empty
    if (field.tagName === "SELECT") {
      const err = field.value ? null : `Please select a ${field.name}.`;
      setFieldValidity(field, err);
    }
    return true;
  }

  const err = rule.validate(field.value.trim());
  setFieldValidity(field, err);
  return err === null;
}

/**
 * Attach real-time validation to the prediction form.
 */
function initFormValidation() {
  const form = document.getElementById("prediction-form");
  if (!form) return;

  // Validate individual fields on blur
  form.querySelectorAll("input, select").forEach((field) => {
    field.addEventListener("blur", () => validateField(field));
    field.addEventListener("input", () => {
      if (field.classList.contains("is-invalid")) validateField(field);
    });
  });

  // Full validation on submit
  form.addEventListener("submit", (e) => {
    let valid = true;
    form.querySelectorAll("input, select").forEach((field) => {
      if (!validateField(field)) valid = false;
    });

    if (!valid) {
      e.preventDefault();
      // Scroll to first error
      const firstError = form.querySelector(".is-invalid");
      if (firstError) {
        firstError.scrollIntoView({ behavior: "smooth", block: "center" });
        firstError.focus();
      }
      return;
    }

    // Show loading overlay
    showLoading();
  });
}

/* ─── Loading Overlay ─────────────────────────────────────────── */

function showLoading() {
  const overlay = document.getElementById("loading-overlay");
  if (overlay) overlay.classList.add("active");
}

function hideLoading() {
  const overlay = document.getElementById("loading-overlay");
  if (overlay) overlay.classList.remove("active");
}

/* ─── Probability Gauge ───────────────────────────────────────── */

/**
 * Animate the circular SVG gauge on the result page.
 * The stroke-dashoffset starts at 502 (full circle hidden) and
 * animates to (1 - probability) * 502 to reveal the filled arc.
 *
 * @param {number} probability - Value between 0 and 100
 */
function animateProbGauge(probability) {
  const fill = document.querySelector(".prob-gauge-fill");
  if (!fill) return;

  const circumference = 502;
  const offset = circumference * (1 - probability / 100);

  // Delay slightly so the CSS transition plays after page paint
  requestAnimationFrame(() => {
    setTimeout(() => {
      fill.style.strokeDashoffset = offset;
    }, 200);
  });
}

/* ─── Result Page Counters ────────────────────────────────────── */

/**
 * Animate numeric values counting up from 0 to their target.
 */
function animateCounters() {
  document.querySelectorAll("[data-counter]").forEach((el) => {
    const target   = parseFloat(el.dataset.counter);
    const suffix   = el.dataset.suffix || "";
    const duration = 1200;
    const start    = performance.now();

    function step(now) {
      const elapsed  = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased    = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      const value    = target * eased;

      el.textContent = (Number.isInteger(target) ? Math.round(value) : value.toFixed(1)) + suffix;

      if (progress < 1) requestAnimationFrame(step);
    }

    requestAnimationFrame(step);
  });
}

/* ─── Tooltip Init ────────────────────────────────────────────── */

function initTooltips() {
  if (typeof bootstrap === "undefined") return;
  document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach((el) => {
    new bootstrap.Tooltip(el, { trigger: "hover" });
  });
}

/* ─── Smooth Scroll ───────────────────────────────────────────── */

function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", (e) => {
      const target = document.querySelector(anchor.getAttribute("href"));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });
}

/* ─── Contact Form Mock Submit ────────────────────────────────── */

function initContactForm() {
  const form = document.getElementById("contact-form");
  if (!form) return;

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const btn  = form.querySelector('[type="submit"]');
    const orig = btn.innerHTML;
    btn.innerHTML = '<span class="loading-spinner" style="width:20px;height:20px;border-width:3px;display:inline-block;vertical-align:middle;margin-right:.5rem;"></span> Sending…';
    btn.disabled = true;

    setTimeout(() => {
      btn.innerHTML = '✓ Message Sent!';
      btn.style.background = '#059669';
      setTimeout(() => {
        btn.innerHTML = orig;
        btn.disabled  = false;
        btn.style.background = '';
        form.reset();
        form.querySelectorAll('.is-valid').forEach(el => el.classList.remove('is-valid'));
      }, 3000);
    }, 1500);
  });
}

/* ─── Boot ────────────────────────────────────────────────────── */

document.addEventListener("DOMContentLoaded", () => {
  initScrollAnimations();
  initNavbarScroll();
  initFormValidation();
  initTooltips();
  initSmoothScroll();
  initContactForm();

  // Result page specific
  const gauge = document.getElementById("prob-gauge-value");
  if (gauge) {
    const prob = parseFloat(gauge.dataset.prob || 0);
    animateProbGauge(prob);
    animateCounters();
  }

  // Hide loading on back-button navigation
  window.addEventListener("pageshow", hideLoading);
});
