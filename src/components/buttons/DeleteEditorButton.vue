<template>
  <span @click="deleteEditor">
    <i class="delete-button fa fa-trash"/>
  </span>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import EditorType from '@/EditorType';
import { Getter, Mutation } from 'vuex-class';
import { saveEditor, SaveWithNames } from '@/file/EditorAsJson';
import { EditorModel } from '@/store/editors/types';

@Component
export default class DeleteEditorButton extends Vue {
  @Prop({ required: true }) readonly editorType!: EditorType;
  @Prop({ required: true }) readonly index!: number;
  @Prop({ required: true }) readonly name!: string;
  @Getter('overviewEditor') overviewEditor!: EditorModel;
  @Getter('saveWithNames') saveWithNames!: SaveWithNames;
  @Mutation('deleteEditor') delete!: (arg0: { editorType: EditorType; index: number }) => void;

  private deleteEditor() {
    this.delete({ editorType: this.editorType, index: this.index });

    // Re-save overview after nodes may have been removed
    const overviewEditorSave = saveEditor(this.overviewEditor);
    this.$cookies.set('unsaved-editor-Overview', overviewEditorSave);

    // Remove deleted editor from saved cookies
    this.$cookies.remove(`unsaved-editor-${this.name}`);
    this.$cookies.set('unsaved-project', this.saveWithNames);
  }
}
</script>

<style lang="scss" scoped>
  .delete-button {
    color: var(--foreground);
    background: none;
    padding: 0 4px 1px;
    border-radius: 2px;
    margin: 0 4px;
    &:hover {
      background: var(--red);
    }
  }
</style>
