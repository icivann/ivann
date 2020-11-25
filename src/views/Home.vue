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
import { FilenamesList, ParsedFile } from '@/store/codeVault/types';

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
  @Mutation('loadFiles') loadFiles!: (files: ParsedFile[]) => void;

  created() {
    // Auto-loading
    if (localStorage.getItem('unsaved-project')) {
      const saveWithNames: SaveWithNames = JSON.parse(localStorage.getItem('unsaved-project')!);
      const overviewEditor = JSON.parse(localStorage.getItem('unsaved-editor-Overview')!);
      const modelEditors = saveWithNames.modelEditors.map((name) => JSON.parse(localStorage.getItem(`unsaved-editor-${name}`)!));
      const dataEditors = saveWithNames.dataEditors.map((name) => JSON.parse(localStorage.getItem(`unsaved-editor-${name}`)!));
      this.loadEditors({
        overviewEditor, modelEditors, dataEditors,
      });
      // We reset the view to set the panning and scaling on the current view.
      EditorManager.getInstance().resetView();

      // Auto-Load Code Vault
      if (localStorage.getItem('unsaved-code-vault')) {
        const filenamesList: FilenamesList = JSON.parse(localStorage.getItem('unsaved-code-vault')!);
        const files = filenamesList.filenames.map((filename: string) => JSON.parse(localStorage.getItem(`unsaved-file-${filename}`)!));
        this.loadFiles(files);
      }
    }

    // Set up auto-save every 5 seconds
    setInterval(() => {
      // Update overview editor if required
      this.updateNodeInOverview(this.currEditorModel);

      // Auto-saving, have to save Overview as that may have changed passively
      const currEditorSave = saveEditor(this.currEditorModel);
      const overviewEditorSave = saveEditor(this.overviewEditor);

      localStorage.setItem('unsaved-project', JSON.stringify(this.saveWithNames));
      localStorage.setItem(`unsaved-editor-${this.currEditorModel.name}`, JSON.stringify(currEditorSave));
      localStorage.setItem('unsaved-editor-Overview', JSON.stringify(overviewEditorSave));
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
