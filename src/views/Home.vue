<template>
  <div class="home container-fluid d-flex flex-column">
    <Titlebar />
    <div class="row flex-grow-1">
      <div class="navbar-col">
        <Navbar />
      </div>
      <div class="col d-flex flex-column p-0">
        <Editor v-show="!inCodeVault"/>
        <CodeVault v-show="inCodeVault"/>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import Navbar from '@/components/navbar/Navbar.vue';
import Editor from '@/components/Editor.vue';
import Titlebar from '@/components/Titlebar.vue';
import { Mutation } from 'vuex-class';
import { Save, SaveWithNames } from '@/file/EditorAsJson';
import EditorManager from '@/EditorManager';
import { mapGetters } from 'vuex';
import CodeVault from '@/components/CodeVault.vue';

@Component({
  components: {
    CodeVault,
    Titlebar,
    Navbar,
    Editor,
  },
  computed: mapGetters(['inCodeVault']),
})
export default class Home extends Vue {
  @Mutation('loadEditors') loadEditors!: (save: Save) => void;

  created() {
    // Auto-loading
    if (this.$cookies.isKey('unsaved-project')) {
      const saveWithNames: SaveWithNames = this.$cookies.get('unsaved-project');
      const overviewEditor = this.$cookies.get('unsaved-editor-Overview');
      const modelEditors = saveWithNames.modelEditors.map((name) => this.$cookies.get(`unsaved-editor-${name}`));
      const dataEditors = saveWithNames.dataEditors.map((name) => this.$cookies.get(`unsaved-editor-${name}`));
      const trainEditors = saveWithNames.trainEditors.map((name) => this.$cookies.get(`unsaved-editor-${name}`));
      this.loadEditors(new Save(overviewEditor, modelEditors, dataEditors, trainEditors));
      // We reset the view to set the panning and scaling on the current view.
      EditorManager.getInstance().resetView();
    }
  }
}
</script>

<style scoped>
.home {
  height: 100vh;
  overflow: auto;
}

.navbar-col {
  width: 3rem;
}
</style>
