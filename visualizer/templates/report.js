const searchInput = document.getElementById('search');
const fundsContainer = document.getElementById('funds-container');
const noResults = document.getElementById('no-results');
const funds = document.querySelectorAll('.fund');

searchInput.addEventListener('input', function() {
    const query = this.value.toLowerCase().trim();
    let visibleCount = 0;

    funds.forEach(fund => {
        const name = fund.dataset.name || '';
        const matches = !query || name.includes(query);
        fund.classList.toggle('hidden', !matches);
        if (matches) visibleCount++;
    });

    noResults.style.display = visibleCount === 0 ? 'block' : 'none';
});

function expandAll() {
    funds.forEach(f => f.open = true);
}

function collapseAll() {
    funds.forEach(f => f.open = false);
}

// Keyboard shortcut: / to focus search
document.addEventListener('keydown', e => {
    if (e.key === '/' && document.activeElement !== searchInput) {
        e.preventDefault();
        searchInput.focus();
    }
    if (e.key === 'Escape') {
        searchInput.blur();
        searchInput.value = '';
        searchInput.dispatchEvent(new Event('input'));
    }
});
