<template>
  <div class="right-sidebar h-100">
    <b-tabs content-class="mt-2 ml-2" nav-class="sidebar-nav-tabs">
        <b-tab title="Layers">
          <LayersTab />
        </b-tab>
        <b-tab title="Search">
          <SearchTab />
        </b-tab>
    </b-tabs>
  </div>
</template>

<script>
import { Component, Vue } from 'vue-property-decorator';
import LayersTab from '@/components/LayersTab.vue';
import SearchTab from '@/components/SearchTab.vue';

@Component({
  components: {
    SearchTab,
    LayersTab,
  },
})
export default class RightSidebar extends Vue {
}
</script>

<style scoped>
.right-sidebar {
  background: var(--background-alt);
  color: var(--foreground);
  border-left: 0.08rem solid var(--grey);
}

/* /deep/ is needed to customise Bootstrap Vue */
/deep/ .sidebar-nav-tabs {
  background: var(--background);
  border-bottom: 0.08rem solid var(--grey);
  color: red;
}

/deep/ .sidebar-nav-tabs .nav-link {
  background: var(--background);
  color: var(--foreground);
  padding-top: 1rem;
  padding-bottom: 1rem;
  border: 0;
}

/deep/ .sidebar-nav-tabs .nav-link.active {
  background: var(--background);
  color: var(--foreground);
  border-bottom: 0.2rem solid var(--blue);
}
</style>
