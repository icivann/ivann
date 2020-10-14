// Import Baklava
import { BaklavaVuePlugin } from '@baklavajs/plugin-renderer-vue';
import '@baklavajs/plugin-renderer-vue/dist/styles.css';

import '@/assets/scss/style.scss'; // Our style

import Vue from 'vue';
import App from './App.vue';
import router from './router';
import store from './store';

// Import fontawesome
import '@fortawesome/fontawesome-free/css/all.css'; // Fontawesome
import '@fortawesome/fontawesome-free/js/all';

Vue.use(BaklavaVuePlugin);
Vue.config.productionTip = false;

new Vue({
  router,
  store,
  render: (h) => h(App),
}).$mount('#app');
