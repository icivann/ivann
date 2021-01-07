// Import Baklava
import { BaklavaVuePlugin } from '@baklavajs/plugin-renderer-vue';
import '@baklavajs/plugin-renderer-vue/dist/styles.css';

// Import Vue-Cookies
import VueCookies from 'vue-cookies';

// Import fontawesome
import '@fortawesome/fontawesome-free/css/all.css'; // Fontawesome
import '@fortawesome/fontawesome-free/js/all';

// Import Vue-Tour
import VueTour from 'vue-tour';

import '@/assets/scss/style.scss'; // Our style

import Vue from 'vue';
import App from './App.vue';
import router from './router';
import store from './store';

require('vue-tour/dist/vue-tour.css');

Vue.use(BaklavaVuePlugin);
Vue.use(VueCookies);
Vue.use(VueTour);
Vue.config.productionTip = false;

new Vue({
  router,
  store,
  render: (h) => h(App),
}).$mount('#app');
