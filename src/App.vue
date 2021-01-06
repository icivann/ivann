<template>
  <div id="app">
    <router-view/>
    <cookie-law theme="dark-lime"></cookie-law>
    <v-tour name="tour" :steps="steps" :options="options"/>
  </div>
</template>

<script>
import CookieLaw from 'vue-cookie-law-with-type';
import { Component, Vue } from 'vue-property-decorator';
import EditorType from '@/EditorType';

@Component({
  components: {
    CookieLaw,
  },
  data() {
    return {
      options: { enabledButtons: { buttonPrevious: false } },
      steps: [
        {
          target: '[data-v-step="welcome"]',
          content: 'Welcome to <strong>IVANN</strong>! Begin by creating a new model by hovering over the icon and clicking the + sign. This should switch tabs for you.',
        },
        {
          target: '[data-v-step="sidebar-model"]',
          content: 'Create a model using the right sidebar. The model should have at least one input and output node.',
          params: {
            placement: 'left',
          },
          before: () => new Promise((resolve) => {
            if (this.$store.getters.modelEditors.length > 0
              && this.$store.getters.currEditorType === EditorType.MODEL) resolve();
          }),
        },
        {
          target: '[data-v-step="overview"]',
          content: 'Switch back to the overview editor so that you can add your newly created model.',
        },
        {
          target: '[data-v-step="sidebar-overview"]',
          content: 'You can find your model in the right sidebar under \'Model\'.',
          before: () => new Promise((resolve) => {
            if (this.$store.getters.currEditorType === EditorType.OVERVIEW) resolve();
          }),
          params: {
            placement: 'left',
          },
        },
        {
          target: '[data-v-step="data"]',
          content: 'Now that you have added a model node, it is time to create a data node. Hover over the icon and click the + sign.',
        },
        {
          target: '[data-v-step="sidebar-data"]',
          content: 'Use the right sidebar to create the data. Make sure to add an output node.',
          before: () => new Promise((resolve) => {
            if (this.$store.getters.dataEditors.length > 0
              && this.$store.getters.currEditorType === EditorType.DATA) resolve();
          }),
          params: {
            placement: 'left',
          },
        },
        {
          target: '[data-v-step="overview"]',
          content: 'Switch back to the overview editor so that you can add your newly created data node.',
        },
        {
          target: '[data-v-step="sidebar-overview"]',
          content: 'You can find your data node in the right sidebar under \'Data\'.',
          before: () => new Promise((resolve) => {
            if (this.$store.getters.currEditorType === EditorType.OVERVIEW) resolve();
          }),
          params: {
            placement: 'left',
          },
        },
        {
          target: '[data-v-step="sidebar-overview"]',
          content: 'Finish your project by adding training nodes, which can be found under \'Train\'.',
          before: () => new Promise((resolve) => {
            if (this.$store.getters.currEditorType === EditorType.OVERVIEW) resolve();
          }),
          params: {
            placement: 'left',
          },
        },
        {
          target: '[data-v-step="export"]',
          content: 'You can export your project into ready-to-run Python code using the \'Generate Code\' button. You can also save and load existing projects.',
          params: {
            placement: 'bottom',
          },
        },
      ],
    };
  },
})
export default class App extends Vue {
  mounted() {
    this.$tours.tour.start();
  }
}
</script>

<style lang="scss">
#nav a {
  font-weight: bold;
  color: #2c3e50;
}

#nav a.router-link-exact-active {
  color: #42b983;
}

.Cookie {
  background: var(--background) !important;
  padding-top: 0.5em !important;
  padding-bottom: 0.5em !important;
  border-top: var(--grey) 1px solid;
}

.Cookie__button {
  background: var(--blue) !important;
  border-radius: 4px !important;
  padding: 0.5em 3em !important;
  white-space: nowrap;
}

.Cookie__button:hover {
  background: #1B67E0 !important;
}

.v-tour > div[id^='v-step-'] {
  background-color: var(--dark-grey);
  filter: none;
  border: solid 1px var(--foreground);
  .v-step__arrow {
    border: solid 1px var(--foreground);
  }
}
</style>
