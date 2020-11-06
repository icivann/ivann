<template>
  <div>
    Function Stuff goes here
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import Tabs from '@/components/tabs/Tabs.vue';
import Tab from '@/components/tabs/Tab.vue';

@Component({
  components: {
    Tab,
    Tabs,
  },
})
export default class FunctionsTab extends Vue {
}
</script>

<style scoped>
</style>
