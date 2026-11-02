(() => {
  const form = document.getElementById('assessment-form');
  if (!form) return;
  const modes = form.querySelectorAll('input[name="location_mode"]');
  const auto = document.getElementById('auto-location-controls');
  const manual = document.getElementById('manual-location-controls');
  const button = document.getElementById('location-button');
  const status = document.getElementById('location-status');
  const latitude = document.getElementById('latitude');
  const longitude = document.getElementById('longitude');
  const manualLatitude = document.getElementById('manual-latitude');
  const manualLongitude = document.getElementById('manual-longitude');
  const confirmation = document.getElementById('crop-confirmation');
  const confirmed = document.getElementById('location-confirmed');
  const selectedMode = () => form.querySelector('input[name="location_mode"]:checked').value;
  const clearLocation = () => { latitude.value = ''; longitude.value = ''; confirmation.checked = false; confirmed.value = 'false'; };
  const updateMode = () => {
    const mode = selectedMode();
    auto.hidden = mode !== 'auto'; manual.hidden = mode !== 'manual';
    manualLatitude.disabled = mode !== 'manual'; manualLongitude.disabled = mode !== 'manual';
    if (mode !== 'auto') { clearLocation(); status.textContent = ''; }
    if (mode !== 'manual') { manualLatitude.value = ''; manualLongitude.value = ''; }
  };
  modes.forEach(mode => mode.addEventListener('change', updateMode));
  confirmation.addEventListener('change', () => { confirmed.value = confirmation.checked ? 'true' : 'false'; });
  button.addEventListener('click', () => {
    if (!navigator.geolocation) { status.textContent = 'Device location is unavailable. Choose Manual Location or No Location.'; return; }
    status.textContent = 'Requesting your device location…';
    navigator.geolocation.getCurrentPosition(position => {
      latitude.value = position.coords.latitude; longitude.value = position.coords.longitude;
      status.textContent = 'Device location captured. Confirm that you are at the crop or field before submitting.';
    }, () => { clearLocation(); status.textContent = 'Device location was denied or unavailable. Choose Manual Location or No Location.'; }, {enableHighAccuracy: true, timeout: 10000, maximumAge: 0});
  });
  form.addEventListener('submit', event => {
    if (selectedMode() === 'manual') {
      manualLatitude.setCustomValidity(''); manualLongitude.setCustomValidity('');
      const lat = Number(manualLatitude.value), lon = Number(manualLongitude.value);
      if (!Number.isFinite(lat) || lat < -90 || lat > 90) manualLatitude.setCustomValidity('Latitude must be between -90 and 90.');
      if (!Number.isFinite(lon) || lon < -180 || lon > 180) manualLongitude.setCustomValidity('Longitude must be between -180 and 180.');
      if (!manualLatitude.checkValidity() || !manualLongitude.checkValidity()) { event.preventDefault(); form.reportValidity(); return; }
      latitude.value = manualLatitude.value; longitude.value = manualLongitude.value;
    }
    if (selectedMode() === 'none') clearLocation();
  });
  updateMode();
})();
