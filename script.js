/*
 * Interactivity for Natalia's personal portfolio.
 *
 * This script populates course, project and work sections, draws a simple
 * bar chart of courses by category and renders an animated wireframe grid
 * in the hero section. All DOM manipulations take place after the
 * document has loaded. The chart and grid are implemented without
 * external libraries to remain dependency‑free and performant. Bars in
 * the chart can be clicked to filter the course list by category.
 */

// Inject theme stylesheet on every page
(function () {
  var link = document.createElement('link');
  link.rel = 'stylesheet';
  // if style_multi_light.css is at repo root, use a root-relative URL:
  link.href = 'style_multi_light.css';
})();

document.addEventListener('DOMContentLoaded', () => {
  /* Data definitions */
  const courses = [
    { title: 'Artificial Intelligence', code: 'CSC384', category: 'Computer Science', description: 'Search algorithms, reasoning, constraint satisfaction, and machine learning basics.' },
    { title: 'Calculus', code: 'MAT137/MAT235', category: 'Mathematics', description: 'Differential and integral calculus of one and several variables with applications.' },
    { title: 'Classical Mechanics', code: 'PHY254', category: 'Physics', description: 'Newtonian mechanics, Lagrangian and Hamiltonian formalisms and rigid body motion.' },
    { title: 'Computational Physics', code: 'PHY407', category: 'Physics', description: 'Numerical methods and scientific computing applied to physical systems.' },
    { title: 'Computer Science/Programming', code: 'CSC108/CSC148', category: 'Computer Science', description: 'Introduction to programming, data types, recursion and data abstraction.' },
    { title: 'Data Structures & Analysis', code: 'CSC263', category: 'Computer Science', description: 'Design and analysis of data structures and algorithms.' },
    { title: 'Electricity & Magnetism', code: 'PHY250', category: 'Physics', description: 'Electrostatics, magnetostatics, Maxwell’s equations and electromagnetic waves.' },
    { title: 'Electronics Lab', code: 'PHY405', category: 'Physics', description: 'Hands‑on experience building and analyzing electronic circuits.' },
    { title: 'Interactive Computational Media', code: 'CSC318', category: 'Computer Science', description: 'Human‑computer interaction and interactive media design.' },
    { title: 'Linear Algebra', code: 'MAT223', category: 'Mathematics', description: 'Vector spaces, linear transformations, eigenvalues and eigenvectors.' },
    { title: 'Machine Learning', code: 'CSC311', category: 'Computer Science', description: 'Supervised and unsupervised learning, neural networks, optimisation.' },
    { title: 'Mathematical Expression & Reasoning for CS', code: 'CSC165', category: 'Computer Science', description: 'Logical reasoning, proof techniques and algorithmic correctness.' },
    { title: 'Optics', code: 'PHY385', category: 'Physics', description: 'Geometrical and physical optics, interference, diffraction and polarization.' },
    { title: 'Ordinary Differential Equations', code: 'MAT244', category: 'Mathematics', description: 'Linear ODEs, series solutions, Laplace transforms and applications.' },
    { title: 'Probability & Statistics', code: 'STA237', category: 'Mathematics', description: 'Random variables, distributions, estimation and hypothesis testing.' },
    { title: 'Software Design', code: 'CSC207', category: 'Computer Science', description: 'Object‑oriented design principles, design patterns and software architectures.' },
    { title: 'Software Tools & Systems Programming', code: 'CSC209', category: 'Computer Science', description: 'Unix tools, shell programming, memory management and concurrency.' },
    { title: 'Thermal Physics', code: 'PHY252', category: 'Physics', description: 'Thermodynamics, statistical mechanics and kinetic theory.' },
    { title: 'Time Series Analysis', code: 'PHY408', category: 'Physics', description: 'Fourier analysis, stochastic processes and time series modelling.' },
    { title: 'Quantum Information', code: 'PHY365', category: 'Physics', description: 'Qubits, quantum gates, entanglement and quantum algorithms.' },
    { title: 'Quantum Mechanics', code: 'PHY256', category: 'Physics', description: 'Wave functions, Schrödinger equation and angular momentum.' },
  ];

  const projects = [
    {
      title: 'Machine Learning for Stress Prediction Using Oura Ring Data',
      description: 'A machine learning project exploring sleep and activity data collected from an Oura ring to predict daily readiness and identify patterns.',
      tags: ['Machine Learning', 'Python', 'Pandas'],
      link: 'projects/oura-ring-ml-analysis.html',
      image: null
    },
    {
      title: 'Predicting Student Performance with Machine Learning',
      description: 'This project explores predicting student responses...',
      tags: ['Machine Learning', 'kNN', 'IRT', 'Autoencoders'],
      link: 'projects/csc311-ml-project.html'
    },
    {
      title: 'Virtual Fitness Assistant',
      description: 'A Java-based program that tracks macro/micro nutrients, evaluates recipes and suggests meals based on dietary and fitness goals.',
      tags: ['Java', 'Nutrition', 'API'],
      image: null
    },
    {
      title: 'Quantum Mechanics Simulator',
      description: 'An educational tool simulating 1D quantum wave functions in potential wells. Visualises probability distributions and energy levels.',
      tags: ['Physics', 'Numerical Methods'],
      link: 'projects/quantum-mechanics-simulator.html',
      image: null
    }
  ];


  const workHistory = [
    {
      role: 'Scientific Programmer',
      company: 'Environment Canada',
      period: 'June 2023 – December 2024',
      description: 'Developed data pipelines and visual analytics tools for research projects spanning physics and biology. Collaborated with researchers to transform raw data into actionable insights.',
      tags: ['Python', 'Java', 'C++']
    },
    {
      role: 'Teaching Assistant',
      company: 'University of Toronto',
      period: '2023 – 2024',
      description: 'Assisted in the delivery of CSC148 (Introduction to Computer Science), holding office hours, marking assignments and running lab sessions.',
      tags: ['Teaching', 'Python']
    }
  ];

  const creativeWorks = [
    {
      title: "Aston Martin F1 Car",
      description: "Colored pencil drawing of the Aston Martin AMR21 Formula 1 car.",
      tags: ["Drawing", "F1", "Colored Pencil"],
      image: "images/creative/F1 cars/aston_martin.jpg"
    },
    {
      title: "Ferrari F2007",
      description: "Detailed illustration of the iconic Ferrari F2007 Formula 1 car.",
      tags: ["Drawing", "F1", "Ferrari"],
      image: "images/creative/F1 cars/ferrari_f2007.jpg"
    },
    {
      title: "Electric Guitar",
      description: "A Les Paul–style electric guitar in vibrant colors.",
      tags: ["Drawing", "Music", "Colored Pencil"],
      image: "images/creative/Guitar/guitar.jpeg"
    },
    {
      title: "Monstera Leaf",
      description: "A botanical drawing of a Monstera Deliciosa with process shots.",
      tags: ["Drawing", "Botanical", "Colored Pencil"],
      image: "images/creative/Monstera Deliciosa/monstera.jpeg"
    },
    {
      title: "Mark Knopfler",
      description: "Work-in-progress piece of Mark Knopfler playing guitar.",
      tags: ["Drawing", "Portrait", "Music"],
      image: "images/creative/Mark Knopfler/MK_1.jpeg"
    },
    {
      title: "Tiger",
      description: "Hyperrealistic colored pencil drawing of a tiger, step-by-step process included.",
      tags: ["Drawing", "Wildlife", "Colored Pencil"],
      image: "images/creative/tiger/tiger.jpeg"
    }
  ];

  function renderCreative() {
    const container = document.getElementById('creative-list');
    container.innerHTML = '';
    creativeWorks.forEach(work => {
      const slug = work.title.toLowerCase().replace(/[^a-z0-9]+/g, '-');
      const card = document.createElement('div');
      card.className = 'card';

      if (work.image) {
        const img = document.createElement('img');
        img.src = work.image;
        img.alt = work.title;
        img.style.width = '100%';
        img.style.borderRadius = 'var(--radius)';
        img.style.marginBottom = '0.8rem';
        card.appendChild(img);
      }

      const title = document.createElement('h3');
      title.textContent = work.title;
      const desc = document.createElement('p');
      desc.textContent = work.description;

      const tagsEl = document.createElement('div');
      tagsEl.className = 'tags';
      work.tags.forEach(t => {
        const span = document.createElement('span');
        span.className = 'tag';
        span.textContent = t;
        tagsEl.appendChild(span);
      });

      card.appendChild(title);
      card.appendChild(desc);
      card.appendChild(tagsEl);

      const link = document.createElement('a');
      link.href = `creative/${slug}.html`;
      link.className = 'card-link';
      link.appendChild(card);

      container.appendChild(link);
    });
  }

  // Call this with your other render calls
  renderCreative();

  /* DOM references */
  const coursesContainer = document.getElementById('courses-list');
  const projectsContainer = document.getElementById('projects-list');
  const workContainer = document.getElementById('work-list');
  const chartSvg = document.getElementById('courseChart');

  let currentFilter = null;

  /* Render functions */
  /**
   * Render the list of courses. Each course is wrapped in an anchor tag
   * that links to a dedicated detail page in the `courses` folder. The
   * slug for each course is derived from its code by replacing any
   * non‑alphanumeric characters with hyphens. When a filter is active
   * (e.g. after clicking a bar in the chart), only courses in that
   * category are displayed.
   */
  function renderCourses(filter = null) {
    coursesContainer.innerHTML = '';
    const filtered = filter ? courses.filter(c => c.category === filter) : courses;
    filtered.forEach(course => {
      // Compute a filename slug based on the course code
      const slug = course.code.replace(/[^A-Za-z0-9]+/g, '-');
      // Card element containing title, description and category tag
      const card = document.createElement('div');
      card.className = 'card';
      const title = document.createElement('h3');
      title.textContent = `${course.title} (` + course.code + ')';
      const desc = document.createElement('p');
      desc.textContent = course.description;
      const tagsEl = document.createElement('div');
      tagsEl.className = 'tags';
      const tag = document.createElement('span');
      tag.className = 'tag';
      tag.textContent = course.category;
      tagsEl.appendChild(tag);
      card.appendChild(title);
      card.appendChild(desc);
      card.appendChild(tagsEl);
      // Wrap the card in an anchor so it acts like a button
      const link = document.createElement('a');
      link.href = `courses/${slug}.html`;
      link.className = 'card-link';
      link.appendChild(card);
      coursesContainer.appendChild(link);
    });
  }

  /**
   * Render the list of projects. Each project card becomes an anchor
   * linking to its own detail page inside the `projects` folder. The
   * slug is derived from the project title by lowercasing and
   * replacing non‑alphanumerics with hyphens. Images, if provided,
   * appear at the top of the card.
   */
  function slugify(text) {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '');
  }

  function renderProjects() {
    projectsContainer.innerHTML = '';
    projects.forEach(project => {
      const slug = slugify(project.title);

      const card = document.createElement('div');
      card.className = 'card';

      // Optional cover image
      if (project.image) {
        const img = document.createElement('img');
        img.src = project.image;
        img.alt = project.title;
        img.style.width = '100%';
        img.style.borderRadius = 'var(--radius)';
        img.style.marginBottom = '0.8rem';
        card.appendChild(img);
      }

      // Title
      const title = document.createElement('h3');
      title.textContent = project.title;

      // Description
      const desc = document.createElement('p');
      desc.textContent = project.description;

      // Tags
      const tagsEl = document.createElement('div');
      tagsEl.className = 'tags';
      project.tags.forEach(t => {
        const span = document.createElement('span');
        span.className = 'tag';
        span.textContent = t;
        tagsEl.appendChild(span);
      });

      card.appendChild(title);
      card.appendChild(desc);
      card.appendChild(tagsEl);

      // Wrap card in anchor
      const link = document.createElement('a');
      // Use explicit link if provided, otherwise fallback to slug
      link.href = project.link || `projects/${slug}.html`;
      link.className = 'card-link';
      link.appendChild(card);

      projectsContainer.appendChild(link);
    });
  }


  function renderWork() {
    workContainer.innerHTML = '';
    workHistory.forEach(job => {
      const card = document.createElement('div');
      card.className = 'card';
      const title = document.createElement('h3');
      title.textContent = `${job.role} @ ${job.company}`;
      const period = document.createElement('p');
      period.textContent = job.period;
      const desc = document.createElement('p');
      desc.textContent = job.description;
      const tagsEl = document.createElement('div');
      tagsEl.className = 'tags';
      job.tags.forEach(t => {
        const span = document.createElement('span');
        span.className = 'tag';
        span.textContent = t;
        tagsEl.appendChild(span);
      });
      card.appendChild(title);
      card.appendChild(period);
      card.appendChild(desc);
      card.appendChild(tagsEl);
      workContainer.appendChild(card);
    });
  }

  /* Compute category counts for chart */
  function getCategoryCounts() {
    const counts = {};
    courses.forEach(course => {
      counts[course.category] = (counts[course.category] || 0) + 1;
    });
    return counts;
  }

  /* Draw bar chart using plain SVG */
  function drawCourseChart() {
    const counts = getCategoryCounts();
    const categories = Object.keys(counts);
    const values = categories.map(cat => counts[cat]);
    const maxVal = Math.max(...values);
    // Clear existing contents
    while (chartSvg.firstChild) chartSvg.removeChild(chartSvg.firstChild);
    // Dimensions and margins
    const rect = chartSvg.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    const margin = { top: 20, right: 20, bottom: 40, left: 40 };
    const chartW = width - margin.left - margin.right;
    const chartH = height - margin.top - margin.bottom;
    // Draw axes
    const axisGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    axisGroup.setAttribute('class', 'axis');
    // x axis line
    const xLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    xLine.setAttribute('x1', margin.left);
    xLine.setAttribute('y1', margin.top + chartH);
    xLine.setAttribute('x2', margin.left + chartW);
    xLine.setAttribute('y2', margin.top + chartH);
    xLine.setAttribute('stroke', getComputedStyle(document.documentElement).getPropertyValue('--colour-border'));
    axisGroup.appendChild(xLine);
    // y axis line
    const yLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    yLine.setAttribute('x1', margin.left);
    yLine.setAttribute('y1', margin.top);
    yLine.setAttribute('x2', margin.left);
    yLine.setAttribute('y2', margin.top + chartH);
    yLine.setAttribute('stroke', getComputedStyle(document.documentElement).getPropertyValue('--colour-border'));
    axisGroup.appendChild(yLine);
    chartSvg.appendChild(axisGroup);
    // Bars
    const barGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    const barSpace = chartW / categories.length;
    const barWidth = barSpace * 0.6;
    categories.forEach((cat, i) => {
      const value = counts[cat];
      const barHeight = (value / maxVal) * chartH;
      const x = margin.left + i * barSpace + (barSpace - barWidth) / 2;
      const y = margin.top + chartH - barHeight;
      const rectEl = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rectEl.setAttribute('class', 'bar');
      rectEl.setAttribute('x', x);
      rectEl.setAttribute('y', y);
      rectEl.setAttribute('width', barWidth);
      rectEl.setAttribute('height', barHeight);
      rectEl.style.cursor = 'pointer';
      rectEl.addEventListener('click', () => {
        // Toggle filter
        if (currentFilter === cat) {
          currentFilter = null;
        } else {
          currentFilter = cat;
        }
        // Re‑render chart to update bar highlight
        drawCourseChart();
        renderCourses(currentFilter);
      });
      // highlight if active
      if (currentFilter && currentFilter === cat) {
        rectEl.setAttribute('fill', '#7af3aa');
      }
      barGroup.appendChild(rectEl);
      // Category label
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', x + barWidth / 2);
      label.setAttribute('y', margin.top + chartH + 20);
      label.setAttribute('text-anchor', 'middle');
      label.textContent = cat;
      barGroup.appendChild(label);
      // Value label on top of bar
      const valueLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      valueLabel.setAttribute('x', x + barWidth / 2);
      valueLabel.setAttribute('y', y - 5);
      valueLabel.setAttribute('text-anchor', 'middle');
      valueLabel.textContent = value;
      barGroup.appendChild(valueLabel);
    });
    chartSvg.appendChild(barGroup);
  }

  /* Hero section animated grid */
  function initGridAnimation() {
    const canvas = document.getElementById('gridCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let width, height, dpr;
    function resize() {
      dpr = window.devicePixelRatio || 1;
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.scale(dpr, dpr);
    }
    // create line positions
    const spacing = 40;
    function draw(time) {
      ctx.clearRect(0, 0, width, height);
      const t = time * 0.0008;
      const rows = Math.ceil(height / spacing) + 1;
      const cols = Math.ceil(width / spacing) + 1;
      // Draw horizontal lines with sine wave offset
      ctx.strokeStyle = 'rgba(0, 210, 106, 0.4)';
      ctx.lineWidth = 1;
      for (let i = 0; i < rows; i++) {
        ctx.beginPath();
        for (let j = 0; j < cols; j++) {
          const x = j * spacing;
          const y0 = i * spacing;
          // Apply sine wave distortion based on x and time
          const offset = Math.sin((x * 0.02) + t + i) * 10;
          const y = y0 + offset;
          if (j === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      }
      // Draw vertical lines with cosine wave offset
      ctx.strokeStyle = 'rgba(0, 210, 106, 0.25)';
      for (let j = 0; j < cols; j++) {
        ctx.beginPath();
        for (let i = 0; i < rows; i++) {
          const y = i * spacing;
          const x0 = j * spacing;
          const offset = Math.cos((y * 0.02) + t + j) * 10;
          const x = x0 + offset;
          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      }
      requestAnimationFrame(draw);
    }
    resize();
    window.addEventListener('resize', resize);
    requestAnimationFrame(draw);
  }

  // Initialize all sections
  renderCourses();
  renderProjects();
  renderWork();
  drawCourseChart();
  initGridAnimation();
});