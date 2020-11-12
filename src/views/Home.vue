<template>
  <div class="home container-fluid d-flex flex-column">
    <Titlebar/>
    <div class="row flex-grow-1">
      <div class="navbar-col">
        <Navbar/>
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
import { mapGetters } from 'vuex';
import { Getter, Mutation } from 'vuex-class';
import { Save, saveEditor, SaveWithNames } from '@/file/EditorAsJson';
import EditorManager from '@/EditorManager';
import { EditorModel } from '@/store/editors/types';
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
  @Getter('currEditorModel') currEditorModel!: EditorModel;
  @Getter('overviewEditor') overviewEditor!: EditorModel;
  @Getter('saveWithNames') saveWithNames!: SaveWithNames;
  @Mutation('loadEditors') loadEditors!: (save: Save) => void;
  @Mutation('updateNodeInOverview') readonly updateNodeInOverview!: (cEditor: EditorModel) => void;

  created() {
    // Auto-loading
    if (this.$cookies.isKey('unsaved-project')) {
      const saveWithNames: SaveWithNames = this.$cookies.get('unsaved-project');
      const overviewEditor = this.$cookies.get('unsaved-editor-Overview');
      const modelEditors = saveWithNames.modelEditors.map((name) => this.$cookies.get(`unsaved-editor-${name}`));
      const dataEditors = saveWithNames.dataEditors.map((name) => this.$cookies.get(`unsaved-editor-${name}`));
      const trainEditors = saveWithNames.trainEditors.map((name) => this.$cookies.get(`unsaved-editor-${name}`));
      this.loadEditors({
        overviewEditor, modelEditors, dataEditors, trainEditors,
      });
      // We reset the view to set the panning and scaling on the current view.
      EditorManager.getInstance().resetView();
    }

    // Set up auto-save every 5 seconds
    setInterval(() => {
      // Update overview editor if required
      this.updateNodeInOverview(this.currEditorModel);

      // Auto-saving, have to save Overview as that may have changed passively
      const currEditorSave = saveEditor(this.currEditorModel);
      const overviewEditorSave = saveEditor(this.overviewEditor);

      this.$cookies.set('unsaved-project', this.saveWithNames);
      this.$cookies.set(`unsaved-editor-${this.currEditorModel.name}`, currEditorSave);
      this.$cookies.set('unsaved-editor-Overview', overviewEditorSave);
    }, 5000);
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
